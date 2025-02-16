from django.db import models
from django.conf import settings
from django.utils.text import slugify
from django.core.exceptions import ValidationError
from django.utils import timezone

class Project(models.Model):
    STATUS_PLANNING = 'PLANNING'
    STATUS_IN_PROGRESS = 'IN_PROGRESS'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_ON_HOLD = 'ON_HOLD'
    STATUS_CANCELLED = 'CANCELLED'
    
    STATUS_CHOICES = [
        (STATUS_PLANNING, 'Planning'),
        (STATUS_IN_PROGRESS, 'In Progress'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_ON_HOLD, 'On Hold'),
        (STATUS_CANCELLED, 'Cancelled'),
    ]

    name = models.CharField(
        max_length=200,
        verbose_name='Project Name',
        help_text='Enter the project name (max 200 characters)'
    )
    slug = models.SlugField(
        max_length=200,
        unique=True,
        blank=True,
        verbose_name='URL Slug',
        help_text='Automatically generated URL-friendly identifier'
    )
    description = models.TextField(
        blank=True,
        null=True,
        verbose_name='Project Description',
        help_text='Detailed description of the project'
    )
    start_date = models.DateField(
        default=timezone.now,
        verbose_name='Start Date',
        help_text='Project commencement date'
    )
    end_date = models.DateField(
        verbose_name='End Date',
        help_text='Project completion date'
    )
    responsible = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='projects_managed',
        verbose_name='Project Manager',
        help_text='User responsible for the project'
    )
    team = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        related_name='projects_team',
        blank=True,
        verbose_name='Project Team',
        help_text='Team members working on the project'
    )
    week_number = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        verbose_name='Starting Week',
        help_text='Calendar week number of project start',
        validators=[models.MinValueValidator(1), models.MaxValueValidator(53)]
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PLANNING,
        verbose_name='Project Status',
        help_text='Current status of the project'
    )
    budget = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        null=True,
        blank=True,
        verbose_name='Project Budget',
        help_text='Total allocated budget for the project'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-start_date']
        verbose_name = 'Project'
        verbose_name_plural = 'Projects'
        indexes = [
            models.Index(fields=['start_date', 'end_date']),
            models.Index(fields=['status']),
            models.Index(fields=['responsible']),
        ]
        constraints = [
            models.CheckConstraint(
                check=models.Q(end_date__gte=models.F('start_date')),
                name='end_date_after_start_date'
            ),
            models.CheckConstraint(
                check=models.Q(week_number__gte=1) & models.Q(week_number__lte=53),
                name='valid_week_number'
            )
        ]

    def __str__(self):
        return f"{self.name} - {self.get_status_display()}"

    def clean(self):
        super().clean()
        if self.end_date < self.start_date:
            raise ValidationError({
                'end_date': 'End date must be after start date'
            })

    def save(self, *args, **kwargs):
        # Automatically set week number from start date if not provided
        if not self.week_number:
            self.week_number = self.start_date.isocalendar()[1]
        
        # Generate unique slug from name
        if not self.slug:
            base_slug = slugify(self.name)
            unique_slug = base_slug
            counter = 1
            while Project.objects.filter(slug=unique_slug).exists():
                unique_slug = f"{base_slug}-{counter}"
                counter += 1
            self.slug = unique_slug
        
        super().save(*args, **kwargs)

    @property
    def duration_days(self):
        """Calculate project duration in days"""
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).days + 1  # Inclusive
        return 0

    @property
    def is_active(self):
        """Check if project is currently active"""
        today = timezone.now().date()
        return self.start_date <= today <= self.end_date

    @property
    def progress_percentage(self):
        """Calculate project progress percentage (placeholder)"""
        # Implement your actual progress calculation logic here
        return 0

    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('project-detail', kwargs={'slug': self.slug})
