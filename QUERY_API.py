from flask import Flask, render_template, request, jsonify
from QUERY_DATA import query_rag

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("query", "")
    if user_input:
        response = query_rag(user_input)
        return jsonify({"response": response})
    return jsonify({"response": "Please enter a valid query."})

if __name__ == "__main__":
    app.run(debug=True)
