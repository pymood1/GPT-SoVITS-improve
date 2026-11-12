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

.status-monitor-root {
    margin: 10px 0 14px;
}

.status-monitor-root > div {
    padding: 0 !important;
}

.status-dashboard {
    position: relative;
    overflow: hidden;
    display: grid;
    gap: 10px;
    padding: 12px;
    border: 1px solid rgba(124, 108, 255, 0.22);
    border-radius: 18px;
    background:
        linear-gradient(
            135deg,
            rgba(124, 108, 255, 0.10),
            rgba(64, 210, 255, 0.06) 45%,
            rgba(255, 255, 255, 0.02)
        );
    box-shadow:
        0 14px 35px rgba(15, 23, 42, 0.10),
        inset 0 1px 0 rgba(255, 255, 255, 0.12);
    isolation: isolate;
}

.status-dashboard::before {
    content: "";
    position: absolute;
    z-index: -1;
    inset: -55%;
    background:
        radial-gradient(
            circle at 28% 28%,
            rgba(124, 108, 255, 0.18),
            transparent 25%
        ),
        radial-gradient(
            circle at 72% 62%,
            rgba(37, 199, 235, 0.14),
            transparent 28%
        );
    animation: status-ambient 12s ease-in-out infinite alternate;
    pointer-events: none;
}

.status-dashboard-top {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
}

.status-card,
.status-activity,
.status-vram-strip {
    position: relative;
    min-width: 0;
    border: 1px solid rgba(127, 127, 127, 0.16);
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

.status-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border-radius: 15px;
    transition:
        transform 180ms ease,
        border-color 180ms ease,
        box-shadow 180ms ease;
}

.status-card:hover {
    transform: translateY(-1px);
    border-color: rgba(124, 108, 255, 0.35);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.10);
}

.status-ring {
    --status-value: 0;
    --status-accent: #7c6cff;
    flex: 0 0 auto;
    display: grid;
    place-items: center;
    width: 66px;
    height: 66px;
    padding: 5px;
    border-radius: 999px;
    background:
        conic-gradient(
            var(--status-accent)
            calc(var(--status-value) * 1%),
            rgba(127, 127, 127, 0.15) 0
        );
    box-shadow: 0 0 24px color-mix(in srgb, var(--status-accent) 18%, transparent);
    transition: background 350ms ease;
}

.status-card-cpu .status-ring {
    --status-accent: #40d2ff;
}

.status-card-ram .status-ring {
    --status-accent: #9c7cff;
}

.status-card-gpu .status-ring {
    --status-accent: #58e6a9;
}

.status-ring-inner {
    display: grid;
    place-items: center;
    width: 100%;
    height: 100%;
    border-radius: inherit;
    background: rgba(18, 24, 38, 0.90);
    color: #f8fafc;
    line-height: 1.05;
}

.status-ring-inner strong {
    font-size: 17px;
    letter-spacing: -0.02em;
}

.status-ring-inner span {
    margin-top: 2px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    opacity: 0.62;
}

.status-card-copy {
    display: grid;
    gap: 5px;
    min-width: 0;
}

.status-card-copy strong {
    overflow: hidden;
    font-size: 14px;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.status-card-copy span {
    overflow-wrap: anywhere;
    font-size: 12px;
    opacity: 0.68;
}

.status-vram-strip {
    display: grid;
    grid-template-columns: auto minmax(80px, 1fr) auto;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
}

.status-vram-track,
.status-progress-track {
    overflow: hidden;
    border-radius: 999px;
    background: rgba(127, 127, 127, 0.16);
}

.status-vram-track {
    height: 6px;
}

.status-vram-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #58e6a9, #40d2ff);
    box-shadow: 0 0 14px rgba(64, 210, 255, 0.28);
    transition: width 420ms ease;
}

.status-activity {
    display: grid;
    gap: 10px;
    padding: 13px 14px;
    border-radius: 15px;
}

.status-activity-running {
    border-color: rgba(88, 230, 169, 0.26);
    box-shadow: inset 3px 0 0 rgba(88, 230, 169, 0.88);
}

.status-activity-head,
.status-activity-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    min-width: 0;
}

.status-activity-head > div {
    min-width: 0;
}

.status-activity h3 {
    overflow: hidden;
    margin: 5px 0 0;
    font-size: 15px;
    line-height: 1.25;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.10em;
    opacity: 0.76;
}

.status-badge i {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: #94a3b8;
}

.status-activity-running .status-badge i {
    background: #58e6a9;
    box-shadow: 0 0 0 0 rgba(88, 230, 169, 0.55);
    animation: status-pulse 1.7s ease-out infinite;
}

.status-progress-number {
    flex: 0 0 auto;
    font-size: 22px;
    letter-spacing: -0.04em;
}

.status-progress-track {
    height: 8px;
}

.status-progress-fill {
    position: relative;
    height: 100%;
    overflow: hidden;
    border-radius: inherit;
    background: linear-gradient(90deg, #7c6cff, #40d2ff, #58e6a9);
    box-shadow: 0 0 18px rgba(64, 210, 255, 0.24);
    transition: width 450ms cubic-bezier(.2, .8, .2, 1);
}

.status-progress-fill::after {
    content: "";
    position: absolute;
    inset: 0;
    width: 40%;
    transform: translateX(-140%) skewX(-18deg);
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.42),
        transparent
    );
    animation: status-shimmer 2.4s ease-in-out infinite;
}

.status-activity-idle .status-progress-fill::after {
    animation: none;
}

.status-activity-meta {
    font-size: 12px;
}

.status-activity-meta strong {
    flex: 0 0 auto;
}

.status-activity-meta span {
    min-width: 0;
    overflow-wrap: anywhere;
    text-align: right;
    opacity: 0.68;
}

@keyframes status-pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(88, 230, 169, 0.48);
    }
    70% {
        box-shadow: 0 0 0 7px rgba(88, 230, 169, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(88, 230, 169, 0);
    }
}

@keyframes status-shimmer {
    0%,
    30% {
        transform: translateX(-140%) skewX(-18deg);
    }
    70%,
    100% {
        transform: translateX(360%) skewX(-18deg);
    }
}

@keyframes status-ambient {
    from {
        transform: translate3d(-2%, -1%, 0) scale(1);
    }
    to {
        transform: translate3d(2%, 1%, 0) scale(1.05);
    }
}

@media (prefers-color-scheme: light) {
    .markdown {
        background-color: lightblue;
        color: #000;
    }

    .status-dashboard {
        background:
            linear-gradient(
                135deg,
                rgba(124, 108, 255, 0.10),
                rgba(64, 210, 255, 0.06) 45%,
                rgba(255, 255, 255, 0.72)
            );
    }

    .status-card,
    .status-activity,
    .status-vram-strip {
        background: rgba(255, 255, 255, 0.70);
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

    .status-dashboard-top {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .status-card-gpu {
        grid-column: 1 / -1;
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

    .status-dashboard {
        padding: 9px;
        border-radius: 15px;
    }

    .status-dashboard-top {
        grid-template-columns: 1fr;
    }

    .status-card-gpu {
        grid-column: auto;
    }

    .status-ring {
        width: 58px;
        height: 58px;
    }

    .status-activity-head,
    .status-activity-meta {
        align-items: flex-start;
    }

    .status-activity-meta {
        flex-direction: column;
        gap: 4px;
    }

    .status-activity-meta span {
        text-align: left;
    }
}

@media (prefers-reduced-motion: reduce) {
    .status-dashboard::before,
    .status-progress-fill::after,
    .status-activity-running .status-badge i {
        animation: none !important;
    }

    .status-card,
    .status-vram-fill,
    .status-progress-fill {
        transition: none !important;
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
