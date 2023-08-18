import datetime
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import openai
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

app = FastAPI()

# Initialize MongoDB client
client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGODB_DB_NAME")]

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIRequest(BaseModel):
    sessionId: str


@app.post("/query")
async def query_openai(request: OpenAIRequest):
    try:
        # Fetch all messages the chat session
        chat_session = await db.chat_sessions.find_one({"_id": request.sessionId})
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        messages = chat_session.get("messages", [])
        # Send messages to OpenAI
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, stream=True
        )

        ai_response = ""

        def event_stream():
            nonlocal ai_response
            for chunk in completion:
                ai_response += chunk.choices[0].delta["content"]
                yield f"data: {chunk.choices[0].delta}\n\n"

        # Store the AI response in MongoDB
        chat_detail = {
            "sessionId": request.sessionId,
            "message": ai_response,
            "timestamp": datetime.datetime.now(),
            "sender": "ai",
        }
        await db.chat_details.insert_one(chat_detail)

        return Response(event_stream(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
