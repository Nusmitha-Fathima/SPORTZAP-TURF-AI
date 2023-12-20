from werkzeug.middleware.dispatcher import DispatcherMiddleware
from welcome import app as flask_app_1
from anomaly import app as flask_app_2
from sentiment import app as flask_app_3
from probability import app as flask_app_4
from income import app as flask_app_5
from booking import app as flask_app_6





application = DispatcherMiddleware(flask_app_1, {
    '/anomaly': flask_app_2,
    '/sentiment': flask_app_3,
    '/prob': flask_app_4,
    '/inco': flask_app_5,
    '/book': flask_app_6,
})
