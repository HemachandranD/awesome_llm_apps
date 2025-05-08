from typing import List, Union, Annotated

from custom_agent import create_agent_executor
from fastapi import FastAPI, Body
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langserve import add_routes

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


class Input(BaseModel):
    chat_history: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        default_factory=list
    )
    input: str


def parse_agent_output(agent_output):
    try:
        return agent_output["output"]
    except Exception as e:
        print("Unable to parse the output: {}", e)


chain = (create_agent_executor() | RunnableLambda(parse_agent_output)).with_types(
    input_type=Input, output_type=str
)


add_routes(
    app,
    chain,
    path="/quiknews",
    playground_type="chat",
    input_type=Annotated[Input, Body()],
    output_type=str
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)