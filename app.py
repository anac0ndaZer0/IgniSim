from ignisim.ui import build_app, BRAND_CSS

app = build_app()

if __name__ == "__main__":
    app.launch(inbrowser=True, css=BRAND_CSS)
