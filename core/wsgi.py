"""
WSGI (Web Server Gateway Interface) Configuration for Production Deployment

This module serves as the primary entry point for production WSGI servers.
It includes comprehensive configuration for:
- Core application initialization
- Advanced middleware components
- Production-specific optimizations
- Security headers configuration
- Monitoring and observability hooks
- Error handling and reporting
- Resource management

Enhanced Features:
- Gzip compression middleware
- Security header injection
- New Relic monitoring integration
- Sentry error tracking
- WhiteNoise static file serving
- Rate limiting protections
- Request/response instrumentation
- Custom middleware examples

Deployment Considerations:
- Designed for containerized environments (Docker/Kubernetes)
- Compatible with major cloud platforms (AWS, GCP, Azure)
- Optimized for load-balanced environments
- Supports zero-downtime deployments
"""

import os
import sys
import time
import signal
import traceback
from importlib import util
from datetime import datetime

# ---------------------------------------------------------------
# 0. Environment Initialization
# ---------------------------------------------------------------
# Critical: Set Django environment first before any other imports
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings.production')

# Add project directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# ---------------------------------------------------------------
# 1. Core Application Configuration
# ---------------------------------------------------------------
from django.core.wsgi import get_wsgi_application
from django.conf import settings

# Initialize base Django application
django_application = get_wsgi_application()

# ---------------------------------------------------------------
# 2. Production Middleware Stack
# ---------------------------------------------------------------
# These middleware components wrap the Django application in reverse order
# (Last middleware will process responses first)

def apply_production_middleware(application):
    """Wrap application in production-grade middleware layers"""
    
    # Security headers middleware
    if util.find_spec('django.middleware.security') is not None:
        from django.middleware.security import SecurityMiddleware
        application = SecurityMiddleware(application)
        
    # Gzip compression
    if util.find_spec('django.middleware.gzip') is not None:
        from django.middleware.gzip import GZipMiddleware
        application = GZipMiddleware(
            application,
            compress_level=6,
            minimum_size=512
        )
    
    # WhiteNoise static files (must come after SecurityMiddleware)
    if settings.SERVE_STATIC_FILES:
        from whitenoise import WhiteNoise
        application = WhiteNoise(
            application,
            root=settings.STATIC_ROOT,
            prefix=settings.STATIC_URL,
            index_file=True,
            autorefresh=settings.DEBUG
        )
    
    # Rate limiting
    if settings.ENABLE_RATE_LIMITING:
        from django_ratelimit.middleware import RatelimitMiddleware
        application = RatelimitMiddleware(application)
    
    # Custom instrumentation middleware
    application = InstrumentationMiddleware(application)
    
    return application

# ---------------------------------------------------------------
# 3. Monitoring & Observability
# ---------------------------------------------------------------
# Initialize APM tools before application starts
if settings.ENABLE_MONITORING:
    try:
        # New Relic monitoring
        if 'newrelic' in settings.INSTALLED_APPS:
            import newrelic.agent
            newrelic.agent.initialize(
                os.path.join(project_root, 'newrelic.ini'),
                settings.ENVIRONMENT
            )
        
        # Sentry error tracking
        if 'sentry' in settings.INSTALLED_APPS:
            import sentry_sdk
            from sentry_sdk.integrations.django import DjangoIntegration
            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                integrations=[DjangoIntegration()],
                environment=settings.ENVIRONMENT,
                release=settings.VERSION
            )
    except Exception as e:
        sys.stderr.write(f"Monitoring initialization failed: {str(e)}\n")

# ---------------------------------------------------------------
# 4. Custom Instrumentation Middleware
# ---------------------------------------------------------------
class InstrumentationMiddleware:
    """Custom middleware for performance tracking and request/response logging"""
    
    def __init__(self, application):
        self.application = application
        
    def __call__(self, environ, start_response):
        start_time = time.time()
        request_id = environ.get('HTTP_X_REQUEST_ID', 'none')
        
        # Pre-request processing
        self.log_request(environ, request_id)
        
        def custom_start_response(status, headers, exc_info=None):
            # Add custom headers
            headers.append(('X-Request-ID', request_id))
            headers.append(('Server-Timing', f'start={start_time}'))
            
            # Execute original start_response
            return start_response(status, headers, exc_info)
        
        try:
            result = self.application(environ, custom_start_response)
        except Exception as e:
            self.handle_error(e, environ)
            raise
        
        # Post-response processing
        self.log_response(environ, status, request_id, start_time)
        return result
    
    def log_request(self, environ, request_id):
        """Log incoming request details"""
        sys.stdout.write(
            f"[{datetime.utcnow().isoformat()}] "
            f"REQUEST: {request_id} "
            f"{environ['REQUEST_METHOD']} {environ['PATH_INFO']}\n"
        )
    
    def log_response(self, environ, status, request_id, start_time):
        """Log response metrics"""
        duration = (time.time() - start_time) * 1000
        sys.stdout.write(
            f"[{datetime.utcnow().isoformat()}] "
            f"RESPONSE: {request_id} "
            f"Status: {status.split(' ')[0]} "
            f"Duration: {duration:.2f}ms\n"
        )
    
    def handle_error(self, exception, environ):
        """Central error handling"""
        traceback.print_exc()
        if settings.ENABLE_ERROR_REPORTING:
            self.report_error(exception, environ)

# ---------------------------------------------------------------
# 5. Resource Management
# ---------------------------------------------------------------
def handle_graceful_shutdown(signum, frame):
    """Handle SIGTERM/SIGINT signals for graceful shutdown"""
    sys.stdout.write("Initiating graceful shutdown...\n")
    
    # Close database connections
    from django.db import connections
    for conn in connections.all():
        conn.close()
    
    # Release other resources
    if hasattr(settings, 'RESOURCE_MANAGER'):
        settings.RESOURCE_MANAGER.release_all()
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_graceful_shutdown)
signal.signal(signal.SIGINT, handle_graceful_shutdown)

# ---------------------------------------------------------------
# 6. Application Composition
# ---------------------------------------------------------------
# Apply middleware stack to Django application
application = apply_production_middleware(django_application)

# ---------------------------------------------------------------
# 7. Cloud Platform Hooks
# ---------------------------------------------------------------
# AWS Lambda support
if 'AWS_EXECUTION_ENV' in os.environ:
    from django_lambda import LambdaHandler
    application = LambdaHandler(application)

# Google Cloud Run support
if 'K_SERVICE' in os.environ:
    from google.cloud import logging
    logging_client = logging.Client()
    logging_client.setup_logging()

# ---------------------------------------------------------------
# 8. Development/Production Guards
# ---------------------------------------------------------------
if settings.DEBUG:
    sys.stdout.write("WARNING: Running in DEBUG mode - not suitable for production!\n")
    
    # Add development-only middleware
    from werkzeug.middleware.profiler import ProfilerMiddleware
    application = ProfilerMiddleware(
        application,
        profile_dir='/tmp/profiles',
        restrictions=[30]
    )

# ---------------------------------------------------------------
# 9. Final Application Export
# ---------------------------------------------------------------
# The final WSGI application should be named 'application'
# for compatibility with WSGI servers

# Optional: UWSGI module export
try:
    from uwsgidecorators import postfork
    @postfork
    def reconnect_resources():
        """Reconnect to resources after uWSGI fork"""
        from django.db import connections
        for conn in connections.all():
            conn.close()
except ImportError:
    pass

# ---------------------------------------------------------------
# 10. Health Check Endpoint (Bypass middleware)
# ---------------------------------------------------------------
def health_check(environ, start_response):
    """Bare-metal health check endpoint"""
    status = '200 OK'
    response_headers = [('Content-type', 'text/plain')]
    start_response(status, response_headers)
    return [b'OK']

# Mount at /healthz (common Kubernetes health check endpoint)
if settings.ENABLE_LIVENESS_PROBES:
    from werkzeug.wsgi import DispatcherMiddleware
    application = DispatcherMiddleware(application, {
        '/healthz': health_check
    })
