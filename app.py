import os
from ignisim.ui import build_app, BRAND_CSS

app = build_app()

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=BRAND_CSS,
        ssr_mode=False
    )
