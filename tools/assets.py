js = """
function deleteTheme() {

const params = new URLSearchParams(window.location.search);
if (params.has('__theme')) {
    params.delete('__theme');
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.location.replace(newUrl);
}

}
"""

css = """
/* Keep app-level selectors under our own classes where possible so the
   layout survives Gradio 4.x/5.x DOM changes. */
.markdown {
    padding: 6px 10px;
    border-radius: 8px;
    overflow-wrap: anywhere;
}

.app-header {
    width: 100%;
    text-align: center;
}

.app-header-notice {
    margin-bottom: 8px;
    font-size: 15px;
    line-height: 1.45;
    overflow-wrap: anywhere;
}

.app-header-links {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
    align-items: center;
    width: 100%;
}

.app-header-links a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    max-width: 100%;
}

.app-header-links img {
    display: block;
    width: auto;
    height: 30px;
    max-width: 100%;
}

@media (prefers-color-scheme: light) {
    .markdown {
        background-color: lightblue;
        color: #000;
    }
}

@media (prefers-color-scheme: dark) {
    .markdown {
        background-color: #4b4b4b;
        color: rgb(244, 244, 245);
    }
}

@media (max-width: 900px) {
    .markdown {
        padding: 6px 8px;
    }

    .app-header-links {
        gap: 8px;
    }

    .app-header-links img {
        height: 26px;
    }
}

@media (max-width: 600px) {
    .app-header-links {
        justify-content: flex-start;
    }

    .app-header-links a {
        flex: 1 1 auto;
    }

    .app-header-links img {
        height: 24px;
    }
}

::selection {
    background: #ffc078 !important;
}

footer {
    height: 50px !important;
    background-color: transparent !important;
    display: flex;
    justify-content: center;
    align-items: center;
}

footer * {
    display: none !important;
}

"""

top_html = """
<div class="app-header">
    <div class="app-header-notice">{}</div>
    <div class="app-header-links">
        <a href="https://github.com/RVC-Boss/GPT-SoVITS" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-GPT--SoVITS-blue.svg?style=for-the-badge&logo=github" style="width: auto; height: 30px;">
        </a>
        <a href="https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e" target="_blank">
            <img src="https://img.shields.io/badge/简体中文-阅读文档-blue?style=for-the-badge&logo=googledocs&logoColor=white" style="width: auto; height: 30px;">
        </a>
        <a href="https://lj1995-gpt-sovits-proplus.hf.space/" target="_blank">
            <img src="https://img.shields.io/badge/免费在线体验-free_online_demo-yellow.svg?style=for-the-badge&logo=huggingface" style="width: auto; height: 30px;">
        </a>
        <a href="https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e" target="_blank">
            <img src="https://img.shields.io/badge/English-READ%20DOCS-blue?style=for-the-badge&logo=googledocs&logoColor=white" style="width: auto; height: 30px;">
        </a>
        <a href="https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE" target="_blank">
            <img src="https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge&logo=opensourceinitiative" style="width: auto; height: 30px;">
        </a>
    </div>
</div>
"""
