from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette_graphene3 import GraphQLApp, make_graphiql_handler

from back.schema import schema

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_route(
    "/graphql",
    GraphQLApp(schema=schema, on_get=make_graphiql_handler()),
)
