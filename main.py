from fastapi import Depends, FastAPI

from app.blogger.model import Generator
from app.blogger.pexels import get_pexels_image
from app.search_engine import get_trends, search as do_search
from app.auth.jwt import get_query_value

# str = Depends(get_query_value)

app = FastAPI()


@app.get("/")
async def root():
    return {}


@app.get("/api/v2/search")
async def search(query):
    return do_search(query)


@app.get("/api/v2/trends")
async def search(query):
    return get_trends(query)


@app.get("/api/v2/search-and-trends")
async def search(query):
    return do_search(query) + get_trends(query)


@app.get("/api/v2/create_sections")
async def create_sections(prompt):
    return app.generator.generate_blog_headings(prompt)


@app.get("/api/v2/create_blog")
async def create_blog(title, sections):
    images = get_pexels_image(title, amount=len(sections))
    output = []
    image_count = 0
    if images == "No results found":
        image_count = 0
    else:
        image_count = len(images)
    output.append(f"<h1>{title}</h1>\n")
    c = 0
    for section in sections:
        expanded_section = app.generator.generate_blog_section(title, section)
        if c < image_count:
            # output.append(f"<h2>{section}</h2>\n<p>{expanded_section}</p>\n\n")
            output.append(f"<img src={images[c][2]} alt={images[c][0]}>\n")
            c += 1
        output.append(f"<h2>{section}</h2>\n<p>{expanded_section}</p>\n\n")
    return output


@app.on_event("startup")
async def startup_event():
    app.generator = Generator("EleutherAI/gpt-neo-2.7B")
