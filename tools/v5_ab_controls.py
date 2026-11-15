"""Fresh acoustic instances and explicit precision controls for diagnostics."""
import ast
import contextlib
import copy
import inspect
import textwrap


def parameter_profile(module):
    groups = {}
    for parameter in module.parameters():
        key = f"{parameter.dtype}@{parameter.device}"
        groups[key] = groups.get(key, 0) + parameter.numel()
    return groups


def runtime_profile(model, vocoder):
    import torch
    return {
        "parameters": {"s2": parameter_profile(model), "cfm": parameter_profile(model.cfm),
                       "vocoder": parameter_profile(vocoder)},
        "autocast_cuda": torch.is_autocast_enabled("cuda"),
        "autocast_cpu": torch.is_autocast_enabled("cpu"),
        "autocast_cuda_dtype": str(torch.get_autocast_dtype("cuda")),
        "autocast_cpu_dtype": str(torch.get_autocast_dtype("cpu")),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cache_inventory": cache_inventory(model),
        "vocoder_cache_inventory": cache_inventory(vocoder),
    }


def cache_inventory(module):
    import torch
    result = {}
    for prefix, child in module.named_modules():
        for key, value in vars(child).items():
            if "cache" not in key.lower() and key not in ("calls", "step"):
                continue
            name = f"{prefix}.{key}"
            if isinstance(value, torch.Tensor):
                result[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
            elif value is None or isinstance(value, (bool, int, float, str)):
                result[name] = value
            else:
                result[name] = {"type": type(value).__name__}
                if hasattr(value, "__len__"):
                    result[name]["length"] = len(value)
    return result


def buffer_snapshot(module):
    return {name: value.detach().cpu().clone() for name, value in module.named_buffers()}


def changed_buffers(module, before):
    import torch
    after = dict(module.named_buffers())
    def equal(a, b):
        b = b.detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.is_floating_point() or a.is_complex():
            return torch.allclose(a, b, rtol=0, atol=0, equal_nan=True)
        return torch.equal(a, b)
    return sorted(name for name in set(before) | set(after)
                  if name not in before or name not in after
                  or not equal(before[name], after[name]))


@contextlib.contextmanager
def precision_context(fp32):
    import torch
    if not fp32:
        yield
        return
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


@contextlib.contextmanager
def initial_noise_override(cfm, noise):
    """Replace ONLY the initial x assignment, preserving the CFM equations.

    FP32 receives the measured current-precision scaled noise as exact FP32
    values. A seed alone is insufficient: randn(fp16) and randn(fp32) differ.
    The override is instance-local; neither torch.randn nor a class is patched.
    """
    import torch
    if noise is None:
        yield
        return
    # CFMV5 delegates to CFM. Extract the owning base method without changing it.
    owner = next(cls for cls in type(cfm).__mro__ if cls.__name__ == "CFM")
    method = inspect.unwrap(owner.inference)
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    fn = tree.body[0]
    fn.decorator_list = []
    count = 0
    body = []
    for statement in fn.body:
        body.append(statement)
        if isinstance(statement, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "x" for t in statement.targets
        ) and any(isinstance(n, ast.Attribute) and n.attr == "randn" for n in ast.walk(statement)):
            body.append(ast.Assign([ast.Name("x", ast.Store())], ast.Call(
                ast.Name("_ab_fixed_noise", ast.Load()), [ast.Name("x", ast.Load())], [])))
            count += 1
    if count != 1:
        raise ValueError("Cannot identify the unique CFM initial-noise assignment")
    fn.body = body
    ns = dict(method.__globals__)
    def fixed(x):
        if x.shape != noise.shape:
            raise ValueError("Matched-noise shape differs")
        return noise.to(device=x.device, dtype=x.dtype).clone()
    ns["_ab_fixed_noise"] = fixed
    exec(compile(ast.fix_missing_locations(tree), "<CFM matched initial noise>", "exec"), ns)
    saved = cfm.__dict__.get("inference")
    cfm.inference = torch.inference_mode()(ns[fn.name]).__get__(cfm, type(cfm))
    try:
        yield
    finally:
        if saved is None:
            del cfm.inference
        else:
            cfm.inference = saved


def _nonpersistent_buffer_dtypes(module):
    """Record runtime dtypes omitted from state_dict (for example DiT freqs_cis)."""
    result = {}
    for prefix, child in module.named_modules():
        for key in child._non_persistent_buffers_set:
            value = child._buffers.get(key)
            if value is None:
                continue
            name = f"{prefix}.{key}" if prefix else key
            result[name] = value.dtype
    return result


def _align_nonpersistent_buffer_dtypes(module, expected):
    """Match runtime-only buffer dtypes without copying warmed/cache values."""
    for prefix, child in module.named_modules():
        for key in child._non_persistent_buffers_set:
            value = child._buffers.get(key)
            if value is None:
                continue
            name = f"{prefix}.{key}" if prefix else key
            dtype = expected.get(name)
            if dtype is None:
                continue
            if value.is_floating_point() or value.is_complex():
                child._buffers[key] = value.to(dtype=dtype)


class FreshAcousticFactory:
    """Reconstruct one CPU-born pair at a time; never deepcopy CUDA modules."""
    def __init__(self, namespace, root):
        import torch
        self.ns, self.root = namespace, root
        self.model, self.vocoder = namespace["vq_model"], namespace["hifigan_model"]
        self.device = namespace["device"]
        self.current_s2 = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.current_vocoder = {k: v.detach().cpu().clone() for k, v in self.vocoder.state_dict().items()}
        # state_dict excludes persistent=False buffers. V5 DiT uses one for
        # TextEmbedding.freqs_cis; the live WebUI's model.half() casts it, while
        # a fresh constructor leaves it FP32 unless we explicitly align it.
        self.current_s2_nonpersistent_dtypes = _nonpersistent_buffer_dtypes(self.model)
        self.current_vocoder_nonpersistent_dtypes = _nonpersistent_buffer_dtypes(self.vocoder)
        self.s2 = namespace["load_sovits_new"](namespace["sovits_path_global"])["weight"]
        self.voc = torch.load(root / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth",
                              map_location="cpu", weights_only=True)
        self.flags = {key: copy.deepcopy(getattr(self.model.cfm, key)) for key in
                      ("noise_temperature", "use_static_cache", "use_conditioner_cache",
                       "use_step_embedding", "cfg_drop_text", "dbcache_spec")}
        # Free the live acoustic pair's GPU memory, restore it in close().
        self.model.cpu()
        self.vocoder.cpu()

    def __call__(self, fp32=False):
        ns, hps = self.ns, self.ns["hps"]
        model = ns["SynthesizerTrnV3"](
            hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers, **hps.model)
        if not hasattr(self.model, "enc_q") and hasattr(model, "enc_q"):
            del model.enc_q
        vocoder = ns["Generator"](
            initial_channel=100, resblock="1", resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[10, 6, 2, 2, 2], upsample_initial_channel=512,
            upsample_kernel_sizes=[20, 12, 4, 4, 4], gin_channels=0, is_bias=True)
        vocoder.remove_weight_norm()
        # assign=True preserves the actual mixed/current parameter dtypes.
        if fp32:
            required = set(model.state_dict())
            missing = required - set(self.s2)
            if missing:
                raise ValueError(f"FP32 checkpoint lacks active S2 keys: {sorted(missing)[:10]}")
            model.load_state_dict({k: self.s2[k] for k in required}, strict=True, assign=True)
        else:
            model.load_state_dict(self.current_s2, strict=True, assign=True)
        vocoder.load_state_dict(self.voc if fp32 else self.current_vocoder, strict=True, assign=True)
        if not fp32:
            _align_nonpersistent_buffer_dtypes(model, self.current_s2_nonpersistent_dtypes)
            _align_nonpersistent_buffer_dtypes(vocoder, self.current_vocoder_nonpersistent_dtypes)
        # Clone assigned CPU storage so no pair can mutate the factory baseline.
        import torch
        for module in (model, vocoder):
            for item in module.modules():
                for key, value in item._parameters.items():
                    if value is not None:
                        item._parameters[key] = torch.nn.Parameter(value.detach().clone(), value.requires_grad)
                for key, value in item._buffers.items():
                    if value is not None:
                        item._buffers[key] = value.detach().clone()
        if fp32:
            model.float()
            vocoder.float()
        for key, value in self.flags.items():
            setattr(model.cfm, key, copy.deepcopy(value))
        try:
            return model.eval().to(self.device), vocoder.eval().to(self.device)
        except Exception:
            model.cpu()
            vocoder.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def close(self):
        self.model.to(self.device)
        self.vocoder.to(self.device)
