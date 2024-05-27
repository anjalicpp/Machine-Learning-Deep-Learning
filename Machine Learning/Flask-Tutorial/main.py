from flask import Flask

app=Flask(__name__)

@app.route("/")#decorater
def home():
    return "Hello World"

if __name__ == "__main__":
    app.run(debug=True)

This basic code opens up url with heelo world msg

we can also give html tags and inline styling also
