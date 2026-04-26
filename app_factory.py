
from __future__ import annotations

import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from api.routes import router as rec_router
from smart_db_csv_builder.api.routers import connections, schema, build, jobs

try:
    from swagger_ui_bundle import swagger_ui_path
except ImportError:
    swagger_ui_path = None

BASE_DIR = Path(__file__).parent
FRONTEND_DIST = BASE_DIR / 'frontend' / 'dist'
STATIC_DIR = BASE_DIR / 'static'
load_dotenv(BASE_DIR / '.env')
STATIC_DIR.mkdir(exist_ok=True)
SWAGGER_ASSETS = ('swagger-ui-bundle.js', 'swagger-ui.css')
INLINE_FAVICON = (
    'data:image/png;base64,'
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X2ioAAAAASUVORK5CYII='
)


def _missing_swagger_assets() -> list[str]:
    return [name for name in SWAGGER_ASSETS if not (STATIC_DIR / name).exists()]


def _ensure_swagger_assets() -> list[str]:
    missing_assets = _missing_swagger_assets()
    if not missing_assets or swagger_ui_path is None:
        return missing_assets
    bundle_dir = Path(swagger_ui_path)
    for asset_name in missing_assets:
        source = bundle_dir / asset_name
        target = STATIC_DIR / asset_name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    return _missing_swagger_assets()


def create_app() -> FastAPI:
    app = FastAPI(
        title='Proactive AI Unified Platform',
        description='Unified DB to CSV to model training and recommendation workflow.',
        version='3.0.0',
        docs_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')
    if FRONTEND_DIST.exists():
        app.mount('/assets', StaticFiles(directory=str(FRONTEND_DIST / 'assets')), name='assets')

    app.include_router(rec_router)
    app.include_router(connections.router, prefix='/smart-db-csv/api/connections', tags=['smart-db-connections'])
    app.include_router(schema.router, prefix='/smart-db-csv/api/schema', tags=['smart-db-schema'])
    app.include_router(build.router, prefix='/smart-db-csv/api/build', tags=['smart-db-build'])
    app.include_router(jobs.router, prefix='/smart-db-csv/api/jobs', tags=['smart-db-jobs'])

    @app.get('/health', include_in_schema=False)
    def unified_health():
        return {'status': 'ok'}

    @app.get('/', include_in_schema=False)
    def serve_frontend():
        index = FRONTEND_DIST / 'index.html'
        if index.exists():
            return FileResponse(str(index))
        return {'message': 'Frontend build missing. Run npm install && npm run build in ./frontend'}

    @app.get('/smart-db-csv', include_in_schema=False)
    def serve_frontend_alias():
        index = FRONTEND_DIST / 'index.html'
        if index.exists():
            return FileResponse(str(index))
        return {'message': 'Frontend build missing. Run npm install && npm run build in ./frontend'}

    @app.get('/docs', include_in_schema=False)
    def custom_swagger_ui():
        missing_assets = _ensure_swagger_assets()
        if missing_assets:
            raise HTTPException(status_code=500, detail='Missing Swagger UI static assets: ' + ', '.join(missing_assets))
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f'{app.title} - Swagger UI',
            swagger_js_url='/static/swagger-ui-bundle.js',
            swagger_css_url='/static/swagger-ui.css',
            swagger_favicon_url=INLINE_FAVICON,
            swagger_ui_parameters=app.swagger_ui_parameters,
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            init_oauth=app.swagger_ui_init_oauth,
        )

    if app.swagger_ui_oauth2_redirect_url:
        @app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
        def swagger_ui_redirect():
            return get_swagger_ui_oauth2_redirect_html()

    return app
