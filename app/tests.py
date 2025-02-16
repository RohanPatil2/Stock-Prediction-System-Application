from django.test import TestCase
from django.core.exceptions import ValidationError
from django.utils import timezone
from django.urls import reverse
from django.contrib.auth import get_user_model
from .models import Project

User = get_user_model()

class ProjectModelTestBase(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Create common test data
        cls.user = User.objects.create_user(
            username='projectmanager',
            email='pm@example.com',
            password='testpass123'
        )
        cls.team_member1 = User.objects.create_user(
            username='member1',
            email='member1@example.com',
            password='testpass123'
        )
        cls.team_member2 = User.objects.create_user(
            username='member2',
            email='member2@example.com',
            password='testpass123'
        )

        # Base valid project data
        cls.valid_data = {
            'name': 'AI Research Initiative',
            'start_date': timezone.now().date(),
            'end_date': timezone.now().date() + timezone.timedelta(days=30),
            'responsible': cls.user,
            'budget': 500000.00,
            'status': Project.STATUS_PLANNING
        }

class ProjectModelFieldTests(ProjectModelTestBase):
    def test_project_creation(self):
        project = Project.objects.create(**self.valid_data)
        self.assertEqual(Project.objects.count(), 1)
        self.assertEqual(project.name, 'AI Research Initiative')
        self.assertEqual(project.responsible, self.user)

    def test_name_field(self):
        # Test max length
        max_length = Project._meta.get_field('name').max_length
        self.assertEqual(max_length, 200)

        # Test name is required
        with self.assertRaises(ValidationError):
            Project.objects.create(**{**self.valid_data, 'name': ''})

    def test_slug_field_auto_generation(self):
        project = Project.objects.create(**self.valid_data)
        self.assertTrue(project.slug)
        self.assertEqual(project.slug, slugify(project.name))

    def test_slug_uniqueness(self):
        # Create first project
        Project.objects.create(**self.valid_data)
        
        # Create second project with same name
        project2 = Project.objects.create(**self.valid_data)
        self.assertNotEqual(project2.slug, self.valid_data['name'].lower())
        self.assertIn('-1', project2.slug)

    def test_start_date_default(self):
        field = Project._meta.get_field('start_date')
        self.assertEqual(field.default, timezone.now)

    def test_status_field(self):
        project = Project.objects.create(**self.valid_data)
        self.assertEqual(project.status, Project.STATUS_PLANNING)
        self.assertEqual(
            project.get_status_display(),
            'Planning'
        )

    def test_week_number_auto_calculation(self):
        test_date = timezone.datetime(2023, 12, 25).date()  # Week 52
        project = Project.objects.create(
            **{**self.valid_data, 'start_date': test_date}
        )
        self.assertEqual(project.week_number, 52)

    def test_budget_field(self):
        # Test valid budget
        project = Project.objects.create(**self.valid_data)
        self.assertEqual(project.budget, 500000.00)

        # Test decimal places
        project.budget = 123456.78
        project.save()
        self.assertEqual(project.budget, 123456.78)

    def test_team_many_to_many(self):
        project = Project.objects.create(**self.valid_data)
        project.team.add(self.team_member1, self.team_member2)
        self.assertEqual(project.team.count(), 2)
        self.assertIn(self.team_member1, project.team.all())

class ProjectModelMethodTests(ProjectModelTestBase):
    def test_str_representation(self):
        project = Project.objects.create(**self.valid_data)
        expected_str = f"{project.name} - {project.get_status_display()}"
        self.assertEqual(str(project), expected_str)

    def test_get_absolute_url(self):
        project = Project.objects.create(**self.valid_data)
        expected_url = reverse('project-detail', kwargs={'slug': project.slug})
        self.assertEqual(project.get_absolute_url(), expected_url)

    def test_clean_method_valid_dates(self):
        project = Project(**self.valid_data)
        project.full_clean()  # Should not raise

    def test_clean_method_invalid_dates(self):
        invalid_data = {
            **self.valid_data,
            'start_date': timezone.now().date(),
            'end_date': timezone.now().date() - timezone.timedelta(days=1)
        }
        project = Project(**invalid_data)
        with self.assertRaises(ValidationError) as cm:
            project.full_clean()
        self.assertIn('end_date', cm.exception.message_dict)

    def test_save_method_slug_generation(self):
        project = Project(**self.valid_data)
        project.save()
        self.assertTrue(project.slug.startswith('ai-research-initiative'))

    def test_save_method_week_number(self):
        # Test week number is set from start date
        test_date = timezone.datetime(2024, 6, 15).date()  # Week 24
        project = Project.objects.create(
            **{**self.valid_data, 'start_date': test_date}
        )
        self.assertEqual(project.week_number, 24)

        # Test manual week number preservation
        project.week_number = 50
        project.save()
        self.assertEqual(project.week_number, 50)

class ProjectPropertyTests(ProjectModelTestBase):
    def test_duration_days(self):
        # Test normal duration
        start = timezone.now().date()
        end = start + timezone.timedelta(days=14)
        project = Project.objects.create(
            **{**self.valid_data, 'start_date': start, 'end_date': end}
        )
        self.assertEqual(project.duration_days, 15)  # Inclusive

        # Test same day
        project.end_date = project.start_date
        project.save()
        self.assertEqual(project.duration_days, 1)

    def test_is_active_property(self):
        # Current project
        project = Project.objects.create(
            name='Current Project',
            start_date=timezone.now().date() - timezone.timedelta(days=5),
            end_date=timezone.now().date() + timezone.timedelta(days=5),
            responsible=self.user
        )
        self.assertTrue(project.is_active)

        # Future project
        future_project = Project.objects.create(
            name='Future Project',
            start_date=timezone.now().date() + timezone.timedelta(days=5),
            end_date=timezone.now().date() + timezone.timedelta(days=10),
            responsible=self.user
        )
        self.assertFalse(future_project.is_active)

        # Past project
        past_project = Project.objects.create(
            name='Past Project',
            start_date=timezone.now().date() - timezone.timedelta(days=10),
            end_date=timezone.now().date() - timezone.timedelta(days=5),
            responsible=self.user
        )
        self.assertFalse(past_project.is_active)

    def test_progress_percentage(self):
        project = Project.objects.create(**self.valid_data)
        self.assertEqual(project.progress_percentage, 0)

class ProjectConstraintTests(ProjectModelTestBase):
    def test_end_date_after_start_date_constraint(self):
        invalid_data = {
            **self.valid_data,
            'start_date': timezone.now().date(),
            'end_date': timezone.now().date() - timezone.timedelta(days=1)
        }
        project = Project(**invalid_data)
        with self.assertRaises(ValidationError):
            project.full_clean()
        with self.assertRaises(ValidationError):
            project.save()

    def test_week_number_range_constraint(self):
        # Test valid week numbers
        Project.objects.create(**{**self.valid_data, 'week_number': 1})
        Project.objects.create(**{**self.valid_data, 'week_number': 53})
        
        # Test invalid week numbers
        with self.assertRaises(ValidationError):
            Project(**{**self.valid_data, 'week_number': 0}).full_clean()
            
        with self.assertRaises(ValidationError):
            Project(**{**self.valid_data, 'week_number': 54}).full_clean()

class ProjectRelationTests(ProjectModelTestBase):
    def test_responsible_foreign_key(self):
        project = Project.objects.create(**self.valid_data)
        self.assertEqual(project.responsible, self.user)
        self.assertIn(project, self.user.projects_managed.all())

    def test_team_members_relation(self):
        project = Project.objects.create(**self.valid_data)
        project.team.add(self.team_member1)
        project.team.add(self.team_member2)
        
        self.assertEqual(project.team.count(), 2)
        self.assertIn(project, self.team_member1.projects_team.all())
        self.assertIn(project, self.team_member2.projects_team.all())

class ProjectMetaOptionsTests(ProjectModelTestBase):
    def test_ordering(self):
        # Create projects with different start dates
        Project.objects.create(**{
            **self.valid_data,
            'name': 'Project A',
            'start_date': timezone.now().date() - timezone.timedelta(days=10)
        })
        Project.objects.create(**{
            **self.valid_data,
            'name': 'Project B',
            'start_date': timezone.now().date() - timezone.timedelta(days=5)
        })
        Project.objects.create(**{
            **self.valid_data,
            'name': 'Project C',
            'start_date': timezone.now().date()
        })

        projects = Project.objects.all()
        self.assertEqual(projects[0].name, 'Project C')
        self.assertEqual(projects[1].name, 'Project B')
        self.assertEqual(projects[2].name, 'Project A')

    def test_verbose_names(self):
        self.assertEqual(Project._meta.verbose_name, 'Project')
        self.assertEqual(Project._meta.verbose_name_plural, 'Projects')

    def test_indexes(self):
        indexes = [idx.fields for idx in Project._meta.indexes]
        expected_indexes = [
            ['start_date', 'end_date'],
            ['status'],
            ['responsible']
        ]
        for expected in expected_indexes:
            self.assertIn(expected, indexes)

class ProjectEdgeCaseTests(ProjectModelTestBase):
    def test_long_project_name(self):
        long_name = 'A' * 200
        project = Project.objects.create(**{
            **self.valid_data,
            'name': long_name
        })
        self.assertEqual(project.name, long_name)

    def test_minimal_budget(self):
        project = Project.objects.create(**{
            **self.valid_data,
            'budget': 0.01
        })
        self.assertEqual(project.budget, 0.01)

    def test_large_budget(self):
        large_budget = 9999999999.99
        project = Project.objects.create(**{
            **self.valid_data,
            'budget': large_budget
        })
        self.assertEqual(project.budget, large_budget)

    def test_same_day_start_end(self):
        same_date = timezone.now().date()
        project = Project.objects.create(**{
            **self.valid_data,
            'start_date': same_date,
            'end_date': same_date
        })
        self.assertEqual(project.duration_days, 1)

    def test_leap_year_dates(self):
        leap_date = timezone.datetime(2024, 2, 29).date()
        project = Project.objects.create(**{
            **self.valid_data,
            'start_date': leap_date,
            'end_date': leap_date + timezone.timedelta(days=30)
        })
        self.assertEqual(project.week_number, leap_date.isocalendar()[1])

class ProjectStateTransitionTests(ProjectModelTestBase):
    def test_status_transitions(self):
        project = Project.objects.create(**self.valid_data)
        
        # Valid transitions
        project.status = Project.STATUS_IN_PROGRESS
        project.full_clean()
        project.save()
        
        project.status = Project.STATUS_ON_HOLD
        project.full_clean()
        project.save()
        
        project.status = Project.STATUS_COMPLETED
        project.full_clean()
        project.save()

    def test_invalid_status(self):
        project = Project.objects.create(**self.valid_data)
        project.status = 'INVALID_STATUS'
        with self.assertRaises(ValidationError):
            project.full_clean()

class ProjectQuerySetTests(ProjectModelTestBase):
    def test_active_projects_manager(self):
        # Create test projects
        Project.objects.create(**{
            **self.valid_data,
            'name': 'Active Project',
            'start_date': timezone.now().date() - timezone.timedelta(days=5),
            'end_date': timezone.now().date() + timezone.timedelta(days=5)
        })
        Project.objects.create(**{
            **self.valid_data,
            'name': 'Future Project',
            'start_date': timezone.now().date() + timezone.timedelta(days=5),
            'end_date': timezone.now().date() + timezone.timedelta(days=10)
        })
        Project.objects.create(**{
            **self.valid_data,
            'name': 'Past Project',
            'start_date': timezone.now().date() - timezone.timedelta(days=10),
            'end_date': timezone.now().date() - timezone.timedelta(days=5)
        })

        # Custom manager method needed (not shown in original model)
        # active_projects = Project.objects.active()
        # self.assertEqual(active_projects.count(), 1)
        # self.assertEqual(active_projects.first().name, 'Active Project')

class ProjectSignalTests(ProjectModelTestBase):
    # Add tests for any custom signals or post-save actions here
    pass
