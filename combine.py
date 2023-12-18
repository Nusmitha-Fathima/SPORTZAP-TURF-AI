from werkzeug.middleware.dispatcher import DispatcherMiddleware
from welcome import app as flask_app_1
from probability import app as flask_app_2
from income import app as flask_app_3
from booking import app as flask_app_4

from dummy import app as flask_app_5


application = DispatcherMiddleware(flask_app_1, {
    '/prob': flask_app_2,
    '/inco': flask_app_3,
    '/book': flask_app_4,
    '/sentiment': flask_app_5,
})
