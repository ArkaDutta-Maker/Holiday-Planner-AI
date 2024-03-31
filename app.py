from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

app = Flask(__name__)

class Travel(BaseModel):
    Sl_No: int = Field(description="Serial Number of the cities")
    City: str = Field(description="List of cities to visit")
    Description: str = Field(description="Description of the cities")

def main(location, distance, crowd, days):
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True,  safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,}, k=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a travel agent who helps in recommending holiday locations. 
        Resident Location is {location}.
        """),
        ("user", "Can you suggest multiple cities to go for a holiday within {distance} and {crowd} crowd for {days} days"),
    ])
    chain = prompt | llm | StrOutputParser()

    output = chain.invoke({"location": location, "distance": distance, "crowd": crowd, "days": days})
    try:
        output_parser = JsonOutputParser(pydantic_object=Travel)
        output_prompt = PromptTemplate(
            template="Think and extract as instructed in JSON format\n{format_instructions}\n{output}\n",
            input_variables=["output"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()},
        )
        final_chain = output_prompt | llm | output_parser
        answer = final_chain.invoke({"output": output})
        return jsonify(answer)
    except:
        return "Error in processing the data"


@app.route('/', methods=['POST'])
def home():
    data = request.get_json()
    
    return main(data['location'], data['distance'], data['crowd'], data['days'])
    

# if __name__ == '__main__':
#     app.run()
