"""Typed-receiver lookups against a real graph: resolved only onto real classes."""

from __future__ import annotations

from .test_inherited_calls import calls_of, index


def test_a_constructed_local_reaches_the_method_in_another_file(writer, graph_db):
    index(writer, {
        "app/forms.py": b"class SignupForm:\n    def clean(self):\n        return 1\n",
        "app/views.py": (b"from app.forms import SignupForm\n\n\n"
                         b"def signup(data):\n    form = SignupForm(data)\n    return form.clean()\n"),
    })
    assert calls_of(graph_db, "app.views.signup") == ["app.forms.SignupForm", "app.forms.SignupForm.clean"]


def test_an_inherited_method_is_found_through_the_type(writer, graph_db):
    index(writer, {
        "app/forms.py": (b"class BaseForm:\n    def clean(self):\n        return 1\n\n\n"
                         b"class SignupForm(BaseForm):\n    pass\n"),
        "app/views.py": (b"from app.forms import SignupForm\n\n\n"
                         b"def signup():\n    form = SignupForm()\n    return form.clean()\n"),
    })
    assert "app.forms.BaseForm.clean" in calls_of(graph_db, "app.views.signup")


def test_a_factory_function_is_not_mistaken_for_a_class(writer, graph_db):
    """`make_form()` returns something; what, the code does not say."""
    index(writer, {
        "app/forms.py": (b"class SignupForm:\n    def clean(self):\n        return 1\n\n\n"
                         b"def make_form():\n    return SignupForm()\n"),
        "app/views.py": (b"from app.forms import make_form\n\n\n"
                         b"def signup():\n    form = make_form()\n    return form.clean()\n"),
    })
    assert calls_of(graph_db, "app.views.signup") == ["app.forms.make_form"]


def test_a_manager_on_an_imported_model(writer, graph_db):
    index(writer, {
        "tags/managers.py": b"class TagManager:\n    def get_by_natural_key(self, name):\n        return name\n",
        "tags/models.py": (b"from tags.managers import TagManager\n\n\n"
                           b"class Tag:\n    objects = TagManager()\n"),
        "tags/views.py": (b"from tags.models import Tag\n\n\n"
                          b"def lookup(name):\n    return Tag.objects.get_by_natural_key(name)\n"),
    })
    # The edge to `Tag` itself is the usage the manager call always made.
    assert calls_of(graph_db, "tags.views.lookup") == [
        "tags.managers.TagManager.get_by_natural_key", "tags.models.Tag"]


def test_a_manager_inherited_from_an_abstract_model(writer, graph_db):
    index(writer, {
        "tags/models.py": (b"class TagManager:\n    def get_by_natural_key(self, name):\n        return name\n\n\n"
                           b"class TagBase:\n    objects = TagManager()\n\n\n"
                           b"class Tag(TagBase):\n    pass\n"),
        "tags/views.py": (b"from tags.models import Tag\n\n\n"
                          b"def lookup(name):\n    return Tag.objects.get_by_natural_key(name)\n"),
    })
    assert calls_of(graph_db, "tags.views.lookup") == [
        "tags.models.Tag", "tags.models.TagManager.get_by_natural_key"]


def test_a_default_django_manager_is_left_unresolved(writer, graph_db):
    """No `objects =` in the repository: Django supplies it, and its methods
    are not the repository's."""
    index(writer, {
        "tags/models.py": b"from django.db import models\n\n\nclass Tag(models.Model):\n    pass\n",
        "tags/views.py": (b"from tags.models import Tag\n\n\n"
                          b"def lookup(name):\n    return Tag.objects.get(name=name)\n"),
    })
    assert calls_of(graph_db, "tags.views.lookup") == ["tags.models.Tag"]
