#!/usr/bin/env python
"""
Django's command-line utility for administrative tasks.

This script sets the default Django settings module and delegates
command-line arguments to Django's management commands.
"""

import os
import sys
import logging

def main():
    """
    Run administrative tasks.

    Sets the default settings module and executes Django's command-line interface.
    """
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        logging.error(
            "Couldn't import Django. Ensure it is installed and available on your PYTHONPATH. "
            "Did you forget to activate your virtual environment?"
        )
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and available on your PYTHONPATH? "
            "Did you forget to activate a virtual environment?"
        ) from exc

    # Execute Django management commands using the command-line arguments.
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
