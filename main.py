from fastapi import FastAPI
from routes.query_router import query_router
from mangum import Mangum

app = FastAPI()
app.include_router(query_router)

handler = Mangum(app)
