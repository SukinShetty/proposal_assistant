from flask import Flask, redirect, render_template_string

app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    # Redirect to your Render deployment
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Proposal Assistant</title>
        <meta http-equiv="refresh" content="0; URL='https://proposal-assistant.onrender.com'" />
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f5f5f5;
            }
            .container {
                text-align: center;
                padding: 20px;
                border-radius: 8px;
                background-color: white;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                max-width: 500px;
            }
            h1 {
                color: #333;
            }
            p {
                color: #666;
                margin: 20px 0;
            }
            a {
                display: inline-block;
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 4px;
                font-weight: bold;
            }
            a:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Proposal Assistant</h1>
            <p>Redirecting you to the application...</p>
            <p>If you are not redirected automatically, click the button below:</p>
            <a href="https://proposal-assistant.onrender.com">Go to Proposal Assistant</a>
        </div>
    </body>
    </html>
    """)

# This is for local development
if __name__ == '__main__':
    app.run(debug=True) 