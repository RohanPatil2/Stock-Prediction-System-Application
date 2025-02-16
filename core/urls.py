"""
Stock Analysis Platform URL Configuration

This URL routing system handles all endpoints for:
- Core website functionality
- User management and authentication
- Stock data analysis and prediction
- Portfolio management
- API endpoints (v1 and v2)
- Administrative interfaces
- Third-party integrations
- Documentation and support

Security considerations:
- All sensitive endpoints use HTTPS
- API endpoints include rate limiting
- Admin panel protected with 2FA
- Sensitive operations require CSRF tokens
"""

from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.sitemaps.views import sitemap
from django.views.decorators.cache import cache_page
from django.views.decorators.http import require_http_methods
from rest_framework.schemas import get_schema_view

from app.views import (
    IndexView,
    StockSearchView,
    StockDetailView,
    StockPredictionView,
    PortfolioDashboardView,
    WatchlistManagerView,
    TransactionHistoryView,
    DividendAnalysisView,
    AlertManagerView,
    ReportGeneratorView,
    APIDocumentationView,
    CustomAuthView,
    TwoFactorAuthView,
    DataExportView,
    ThirdPartyIntegrationsView,
    MarketOverviewView,
    SectorAnalysisView,
    EarningsCalendarView,
    InstitutionalHoldingsView,
    ShortInterestView,
    StockScreenerView,
    BacktestStrategyView,
    RiskAnalysisView,
    Custom404View,
    Custom500View,
)

from app.api import (
    StockDataAPIView,
    PortfolioAPIView,
    WatchlistAPIView,
    AlertAPIView,
    UserAPIView,
    MarketDataAPIView,
    FundamentalDataAPIView,
    TechnicalAnalysisAPIView,
    NewsSentimentAPIView,
)

from app.sitemaps import (
    StaticViewSitemap,
    StockSitemap,
    PortfolioSitemap,
    DocumentationSitemap,
)

# Sitemap configuration
sitemaps = {
    'static': StaticViewSitemap,
    'stocks': StockSitemap,
    'portfolios': PortfolioSitemap,
    'docs': DocumentationSitemap,
}

# API schema configuration
api_schema_view = get_schema_view(
    title="Stock Analysis Platform API",
    description="Comprehensive API for financial data analysis and portfolio management",
    version="2.0.0",
    public=True,
)

urlpatterns = [
    # ======================================================================
    # Core Website Functionality
    # ======================================================================
    path('', IndexView.as_view(), name='home'),
    path('market-overview/', MarketOverviewView.as_view(), name='market-overview'),
    path('sector-analysis/', SectorAnalysisView.as_view(), name='sector-analysis'),
    path('earnings-calendar/', EarningsCalendarView.as_view(), name='earnings-calendar'),
    path('institutional-holdings/', InstitutionalHoldingsView.as_view(), name='institutional-holdings'),
    path('short-interest/', ShortInterestView.as_view(), name='short-interest'),
    
    # ======================================================================
    # Stock Analysis & Prediction
    # ======================================================================
    path('search/', StockSearchView.as_view(), name='stock-search'),
    path('stock/<slug:ticker>/', StockDetailView.as_view(), name='stock-detail'),
    path('stock/<slug:ticker>/prediction/<int:days>/', 
         cache_page(60 * 15)(StockPredictionView.as_view()), 
         name='stock-prediction'),
    path('stock-screener/', StockScreenerView.as_view(), name='stock-screener'),
    path('technical-analysis/', TechnicalAnalysisView.as_view(), name='technical-analysis'),
    path('backtest-strategy/', BacktestStrategyView.as_view(), name='backtest-strategy'),
    path('risk-analysis/', RiskAnalysisView.as_view(), name='risk-analysis'),
    
    # ======================================================================
    # Portfolio Management
    # ======================================================================
    path('portfolio/', 
         login_required(PortfolioDashboardView.as_view()), 
         name='portfolio-dashboard'),
    path('watchlist/', 
         login_required(WatchlistManagerView.as_view()), 
         name='watchlist-manager'),
    path('transactions/', 
         login_required(TransactionHistoryView.as_view()), 
         name='transaction-history'),
    path('dividends/', 
         login_required(DividendAnalysisView.as_view()), 
         name='dividend-analysis'),
    path('alerts/', 
         login_required(AlertManagerView.as_view()), 
         name='alert-manager'),
    path('reports/', 
         login_required(ReportGeneratorView.as_view()), 
         name='report-generator'),
    
    # ======================================================================
    # User Management & Authentication
    # ======================================================================
    path('accounts/', include('django.contrib.auth.urls')),
    path('accounts/register/', CustomAuthView.as_view(), name='register'),
    path('accounts/2fa/', TwoFactorAuthView.as_view(), name='two-factor-auth'),
    path('accounts/data-export/', DataExportView.as_view(), name='data-export'),
    path('accounts/integrations/', ThirdPartyIntegrationsView.as_view(), name='third-party-integrations'),
    
    # ======================================================================
    # API Endpoints (Versioned)
    # ======================================================================
    path('api/v1/', include([
        path('stocks/', StockDataAPIView.as_view(), name='api-v1-stocks'),
        path('portfolio/', PortfolioAPIView.as_view(), name='api-v1-portfolio'),
        path('watchlist/', WatchlistAPIView.as_view(), name='api-v1-watchlist'),
        path('alerts/', AlertAPIView.as_view(), name='api-v1-alerts'),
    ])),
    
    path('api/v2/', include([
        path('stocks/', StockDataAPIView.as_view(), name='api-v2-stocks'),
        path('market/', MarketDataAPIView.as_view(), name='api-v2-market'),
        path('fundamentals/', FundamentalDataAPIView.as_view(), name='api-v2-fundamentals'),
        path('technical/', TechnicalAnalysisAPIView.as_view(), name='api-v2-technical'),
        path('news-sentiment/', NewsSentimentAPIView.as_view(), name='api-v2-news-sentiment'),
        path('users/', UserAPIView.as_view(), name='api-v2-users'),
    ])),
    
    # ======================================================================
    # Documentation & Support
    # ======================================================================
    path('documentation/', APIDocumentationView.as_view(), name='api-documentation'),
    path('schema/', api_schema_view, name='api-schema'),
    path('sitemap.xml', sitemap, {'sitemaps': sitemaps}, name='django.contrib.sitemaps.views.sitemap'),
    path('robots.txt', TemplateView.as_view(template_name="robots.txt", content_type="text/plain")),
    path('security.txt', TemplateView.as_view(template_name="security.txt", content_type="text/plain")),
    
    # ======================================================================
    # Administrative Interfaces
    # ======================================================================
    path('admin/', include([
        path('', admin.site.urls),
        path('advanced/', include('advanced_admin.urls')),
        path('reports/', include('admin_reports.urls')),
        path('audit-log/', include('audit_log.urls')),
    ])),
    
    # ======================================================================
    # Third-party Integrations
    # ======================================================================
    path('integrations/', include([
        path('brokerage/', include('brokerage_integration.urls')),
        path('news/', include('news_providers.urls')),
        path('data-feeds/', include('data_feeds.urls')),
    ])),
    
    # ======================================================================
    # Utility & Maintenance Routes
    # ======================================================================
    path('health/', include('health_check.urls')),
    path('maintenance/', TemplateView.as_view(template_name="maintenance.html")),
    path('status/', include('server_status.urls')),
    
    # ======================================================================
    # Error Handling
    # ======================================================================
    path('404/', Custom404View.as_view(), name='404-error'),
    path('500/', Custom500View.as_view(), name='500-error'),
    
    # ======================================================================
    # Development/Debug Routes (Disable in production)
    # ======================================================================
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

if settings.DEBUG:
    urlpatterns += [
        # Debug toolbar
        path('__debug__/', include('debug_toolbar.urls')),
        
        # Test pages
        path('test/error-400/', TemplateView.as_view(template_name="400.html")),
        path('test/error-403/', TemplateView.as_view(template_name="403.html")),
        path('test/error-404/', TemplateView.as_view(template_name="404.html")),
        path('test/error-500/', TemplateView.as_view(template_name="500.html")),
        
        # Style guide
        path('style-guide/', TemplateView.as_view(template_name="style_guide.html")),
    ]

# ======================================================================
# REST Framework Authentication
# ======================================================================
urlpatterns += [
    path('api-auth/', include('rest_framework.urls')),
    path('api-token-auth/', include('rest_framework.authtoken.urls')),
    path('api-jwt-auth/', include('rest_jwt.urls')),
]

# ======================================================================
# Legacy URL Support
# ======================================================================
legacy_urlpatterns = [
    re_path(r'^legacy/stock.php', StockDetailView.as_legacy_view()),
    re_path(r'^legacy/portfolio.cgi', PortfolioDashboardView.as_legacy_view()),
]

urlpatterns += legacy_urlpatterns

# ======================================================================
# Custom Middleware Routes
# ======================================================================
urlpatterns += [
    path('middleware-test/', 
         require_http_methods(["GET"])(TemplateView.as_view(template_name="middleware_test.html")),
         name='middleware-test'),
]

# ======================================================================
# Subscription Management
# ======================================================================
urlpatterns += [
    path('subscriptions/', include([
        path('', login_required(TemplateView.as_view(template_name="subscriptions/main.html")), name='subscriptions'),
        path('plans/', login_required(TemplateView.as_view(template_name="subscriptions/plans.html")), name='subscription-plans'),
        path('billing/', login_required(TemplateView.as_view(template_name="subscriptions/billing.html")), name='billing-info'),
        path('history/', login_required(TemplateView.as_view(template_name="subscriptions/history.html")), name='billing-history'),
    ])),
]

# ======================================================================
# Compliance & Legal
# ======================================================================
urlpatterns += [
    path('legal/', include([
        path('terms/', TemplateView.as_view(template_name="legal/terms.html"), name='terms-of-service'),
        path('privacy/', TemplateView.as_view(template_name="legal/privacy.html"), name='privacy-policy'),
        path('disclaimer/', TemplateView.as_view(template_name="legal/disclaimer.html"), name='disclaimer'),
        path('gdpr/', TemplateView.as_view(template_name="legal/gdpr.html"), name='gdpr-compliance'),
    ])),
]

# ======================================================================
# Community Features
# ======================================================================
urlpatterns += [
    path('community/', include([
        path('forum/', include('forum.urls')),
        path('blog/', include('blog.urls')),
        path('education/', include('education.urls')),
    ])),
]

# ======================================================================
# Webhook Endpoints
# ======================================================================
urlpatterns += [
    path('webhooks/', include([
        path('stripe/', include('payments.webhooks.stripe')),
        path('plaid/', include('brokerage_integration.webhooks.plaid')),
        path('market-data/', include('data_feeds.webhooks.market_data')),
    ])),
]

# ======================================================================
# Internationalization
# ======================================================================
urlpatterns += [
    path('i18n/', include('django.conf.urls.i18n')),
    path('language/', include('locale_middleware.urls')),
]

handler404 = 'app.views.Custom404View'
handler500 = 'app.views.Custom500View'
handler403 = 'app.views.Custom403View'
handler400 = 'app.views.Custom400View'
