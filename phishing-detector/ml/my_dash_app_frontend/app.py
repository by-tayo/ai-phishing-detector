import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import requests

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder="assets"
)

API_URL = "http://127.0.0.1:8000"

url_tab_content = html.Div(className="card", children=[
    html.P("Enter a URL to check if it's phishing or legitimate.",
           className="subtitle"),

    dbc.Row([
        dbc.Col([
            dbc.Input(
                id="url-input",
                type="text",
                placeholder="Enter a URL (e.g. http://suspicious-login.com)",
                size="lg",
                className="mb-3"
            ),
        ], width=9),
        dbc.Col([
            dbc.Select(
                id="model-select",
                options=[
                    {"label": "ML Model (XGBoost)", "value": "predict"},
                    {"label": "Transformer (BERT)", "value": "predict/transformer"},
                    {"label": "Live Check (OpenPhish)", "value": "check-live"},
                ],
                value="predict",
                className="mb-3"
            ),
        ], width=3),
    ]),

    dbc.Button(
        "Check for Phishing",
        id="check-button",
        color="primary",
        size="lg",
        className="w-100 mb-3"
    ),

    dcc.Loading(
        id="loading",
        type="circle",
        children=[html.Div(id="result-output")]
    ),
])

email_tab_content = html.Div(className="card", children=[
    html.P("Paste a raw email (headers + body, e.g. from \"Show Original\") or just the "
           "email body text to check it for phishing.",
           className="subtitle"),

    dcc.Textarea(
        id="email-input",
        placeholder="Paste the email here...",
        className="mb-3",
        style={"width": "100%", "height": "220px"},
    ),

    dbc.Button(
        "Scan Email",
        id="scan-email-button",
        color="primary",
        size="lg",
        className="w-100 mb-3"
    ),

    dcc.Loading(
        id="email-loading",
        type="circle",
        children=[html.Div(id="email-result-output")]
    ),
])

app.layout = html.Div(className="container", children=[
    html.Div(className="card", children=[
        html.H1("🛡️ AI-Powered Phishing Detector"),
    ]),

    dbc.Tabs([
        dbc.Tab(url_tab_content, label="🔗 URL Scanner"),
        dbc.Tab(email_tab_content, label="📧 Email Scanner"),
    ]),

    html.Div(className="card", children=[
        html.H4("📊 How it works"),
        html.P("This tool uses several detection methods:"),
        html.Ul([
            html.Li("🤖 ML Model (XGBoost) — trained on 111,754 URLs with 99.92% accuracy"),
            html.Li("🧠 Transformer (BERT) — fine-tuned DistilBERT with 99.98% accuracy"),
            html.Li("🌐 Live Check — queries OpenPhish live threat database"),
            html.Li("📧 Email Scanner — combines an ML text classifier, sender/header "
                    "heuristics (spoofing, urgency language, mismatched links), and the "
                    "URL model on any links found in the email"),
        ])
    ])
])

@callback(
    Output("result-output", "children"),
    Input("check-button", "n_clicks"),
    State("url-input", "value"),
    State("model-select", "value"),
    prevent_initial_call=True
)
def check_url(n_clicks, url, model):
    if not url:
        return dbc.Alert("Please enter a URL.", color="warning")

    try:
        response = requests.post(
            f"{API_URL}/{model}",
            json={"text": url},
            timeout=10
        )
        result = response.json()

        is_phishing = result.get("is_phishing", False)
        confidence = result.get("confidence", 0)
        confidence_pct = f"{confidence * 100:.2f}%"

        if is_phishing:
            return html.Div(className="phishing", children=[
                html.H3("⚠️ Phishing Detected!"),
                html.P(f"URL: {url}"),
                html.P(f"Confidence: {confidence_pct}"),
                html.P("This URL appears to be malicious. Do not visit it.")
            ])
        else:
            return html.Div(className="safe", children=[
                html.H3("✅ Safe"),
                html.P(f"URL: {url}"),
                html.P(f"Confidence: {confidence_pct}"),
                html.P("This URL appears to be legitimate.")
            ])

    except Exception as e:
        return dbc.Alert(f"Error connecting to API: {str(e)}", color="danger")


@callback(
    Output("email-result-output", "children"),
    Input("scan-email-button", "n_clicks"),
    State("email-input", "value"),
    prevent_initial_call=True
)
def check_email(n_clicks, raw_email):
    if not raw_email:
        return dbc.Alert("Please paste an email to scan.", color="warning")

    try:
        response = requests.post(
            f"{API_URL}/predict/email",
            json={"raw_email": raw_email},
            timeout=15
        )
        result = response.json()

        is_phishing = result.get("is_phishing", False)
        confidence = result.get("confidence", 0)
        confidence_pct = f"{confidence * 100:.2f}%"
        reasons = result.get("reasons", [])
        links = result.get("links", [])
        sender = result.get("sender")
        subject = result.get("subject")

        detail_rows = []
        if sender:
            detail_rows.append(html.P(f"Sender: {sender}"))
        if subject:
            detail_rows.append(html.P(f"Subject: {subject}"))
        detail_rows.append(html.P(f"Confidence: {confidence_pct}"))

        reasons_block = html.Div()
        if reasons:
            reasons_block = html.Div([
                html.P("Why:", className="mb-1", style={"fontWeight": "bold"}),
                html.Ul([html.Li(reason) for reason in reasons], className="reasons-list"),
            ])

        links_block = html.Div()
        if links:
            links_block = html.Div([
                html.P("Links found in the email:", className="mb-1", style={"fontWeight": "bold"}),
                html.Ul([
                    html.Li([
                        html.Span(
                            "⚠️ " if link.get("is_phishing") else "✅ ",
                        ),
                        f"{link.get('url')} — {link.get('confidence', 0) * 100:.2f}% phishing confidence",
                    ], className="link-badge phishing" if link.get("is_phishing") else "link-badge safe")
                    for link in links
                ]),
            ])

        card_class = "phishing" if is_phishing else "safe"
        headline = "⚠️ Phishing Detected!" if is_phishing else "✅ Looks Safe"

        return html.Div(className=card_class, children=[
            html.H3(headline),
            *detail_rows,
            reasons_block,
            links_block,
        ])

    except Exception as e:
        return dbc.Alert(f"Error connecting to API: {str(e)}", color="danger")


if __name__ == "__main__":
    app.run(debug=True, port=8050)
