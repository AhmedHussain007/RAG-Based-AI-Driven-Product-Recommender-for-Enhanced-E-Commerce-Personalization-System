from django.utils.deprecation import MiddlewareMixin
from pyDatalog import pyDatalog

class PyDatalogMiddleware(MiddlewareMixin):
    def process_request(self, request):
        pyDatalog.Logic()
