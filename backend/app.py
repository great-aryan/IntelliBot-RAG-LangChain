# backend/app.py
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
from pymongo import MongoClient
from langchain import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.prompts import ChatPromptTemplate
from rag_model import generate_response

app = Flask(__name__)
CORS(app)
api = Api(app)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["rag_chatbot"]

# Initialize LangChain
prompt_template = ChatPromptTemplate.from_template("The user said: {user_message}\nThe bot should respond with:")
chain = SimpleSequentialChain([LLMChain(prompt_template=prompt_template)])

class Message(Resource):
    def get(self):
        messages = list(db.messages.find())
        return {'messages': messages}, 200

    def post(self):
        user_message = request.json['message']
        db.messages.insert_one({'message': user_message})

        bot_response = chain.run(user_message)
        db.messages.insert_one({'message': bot_response})

        return {'message': bot_response}, 201

api.add_resource(Message, '/message')

if __name__ == '__main__':
    app.run(debug=True)