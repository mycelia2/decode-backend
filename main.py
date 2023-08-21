import datetime
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging
import traceback
from bson.objectid import ObjectId

# Load the .env file
load_dotenv()

app = FastAPI()

# Add CORS middleware
origins = [
    "*",  # Allows all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize MongoDB client
client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGODB_DB_NAME")]

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIRequest(BaseModel):
    sessionId: str
    # latestChatDetail: dict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ai_response = ""


@app.post("/query")
async def query_openai(request: OpenAIRequest):
    try:
        # Fetch the chat session
        print(request.sessionId)
        chat_session = await db["chatsessions"].find_one(
            {"_id": ObjectId(request.sessionId)}  # Use the session ID from the request
        )

        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        # Fetch all chat details for the given session
        chat_details_cursor = db["chatdetails"].find(
            {"sessionId": ObjectId(request.sessionId)}
        )
        messages = [
            {"role": detail["sender"], "content": detail["message"]}
            for detail in await chat_details_cursor.to_list(length=1000)
        ]

        # Send messages to OpenAI and stream the response
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, stream=True
        )

        # Define a generator function to handle the streaming response
        async def event_stream():
            ai_response = ""
            for chunk in completion:
                if "content" in chunk.choices[0].delta:
                    ai_response_chunk = chunk.choices[0].delta["content"]
                    ai_response += ai_response_chunk
                    yield f"data: {ai_response_chunk}"
            # Store the complete AI response in MongoDB
            chat_detail = {
                "sessionId": request.sessionId,
                "message": ai_response,
                "timestamp": datetime.datetime.now(),
                "sender": "ai",
            }
            await db.chat_details.insert_one(chat_detail)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        # Log the error and traceback
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
