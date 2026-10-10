import json
import os
from pathlib import Path

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import requests

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder="assets"
)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
METRICS_PATH = Path(__file__).resolve().parent.parent / "model_metrics.json"


def url_model_summary() -> str:
    try:
        m = json.loads(METRICS_PATH.read_text())
        test = m["held_out_test"]
        with_check = m.get("held_out_test_with_known_domains", {})
        name = m["model"].replace("_", " ").title()
        text = (f"🤖 URL model ({name}): trained on {m['urls']:,} URLs from {m['websites']:,} websites, "
                f"{test['accuracy']:.1%} accuracy on websites it never saw in training")
        if with_check:
            text += f" ({with_check['accuracy']:.1%} with the known-site check)"
        return text
    except (OSError, KeyError, ValueError):
        return "🤖 URL model: run ml/train.py to see its test results here"

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
                    {"label": "ML Model", "value": "predict"},
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
            html.Li(url_model_summary()),
            html.Li("✅ Known-site check — popular sites from the Tranco research list are "
                    "trusted before the model runs; free hosting, link shorteners and "
                    "redirect links never are"),
            html.Li("🌐 Live Check — looks the URL up in the OpenPhish live threat feed"),
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

        if "in_database" in result:  # Live Check returns a lookup, not a prediction
            if result.get("error"):
                return dbc.Alert(f"OpenPhish lookup failed: {result['error']}", color="danger")
            if result["in_database"]:
                return html.Div(className="phishing", children=[
                    html.H3("⚠️ Listed in OpenPhish"),
                    html.P(f"URL: {url}"),
                    html.P("This exact URL is in the OpenPhish live phishing feed. Do not visit it."),
                ])
            return html.Div(className="safe", children=[
                html.H3("ℹ️ Not in OpenPhish"),
                html.P(f"URL: {url}"),
                html.P("This exact URL isn't in the live feed. That doesn't prove it's safe; "
                       "new phishing pages appear every day, so try the ML Model too."),
            ])

        is_phishing = result.get("is_phishing", False)
        confidence = result.get("confidence", 0)
        confidence_pct = f"{confidence * 100:.2f}%"
        reasons = result.get("reasons", [])
        reasons_block = html.Div()
        if reasons:
            reasons_block = html.Div([
                html.P("Why:", className="mb-1", style={"fontWeight": "bold"}),
                html.Ul([html.Li(reason) for reason in reasons], className="reasons-list"),
            ])

        if is_phishing:
            return html.Div(className="phishing", children=[
                html.H3("⚠️ Phishing Detected!"),
                html.P(f"URL: {url}"),
                html.P(f"Confidence: {confidence_pct}"),
                reasons_block,
                html.P("This URL appears to be malicious. Do not visit it.")
            ])
        else:
            return html.Div(className="safe", children=[
                html.H3("✅ Safe"),
                html.P(f"URL: {url}"),
                html.P(f"Confidence: {confidence_pct}"),
                reasons_block,
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
    app.run(debug=os.environ.get("DASH_DEBUG") == "1", port=8050)
