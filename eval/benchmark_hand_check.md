# Where SEI and jedi disagree, checked by hand

`python -m eval.benchmark` takes jedi's callers as the answer. This file records
whether that is right, for 20 disagreements from the run over every function
(django-taggit 6.1.0 `4e5f7de6`, django-filter 26.2 `daddf7b6`, 2026-11-14):

- all 3 callers SEI reports and jedi does not, and
- 17 of the 92 callers jedi reports and SEI does not, drawn with
  `random.Random(0).sample`.

Each was judged by reading the call site in the pinned source.

**Result: jedi was wrong in 3 of 20 — all three of the callers only SEI found.
In all 17 sampled misses, jedi was right.** jedi never reported something that
was not a call. What it missed are calls it does not attribute: inside a
lambda, inside a nested `def` used to build a property, and through `super()`.
So django-filter's "precision, all" of 0.988 understates SEI. The recall figures
stand.

| # | Kind | Caller → function | Call site | Verdict |
|---|---|---|---|---|
| 1 | SEI only | `filters.Filter.label.fget` → `utils.label_for_filter` | `filters.py:138` `self._label = label_for_filter(…)` inside `fget` | **jedi wrong**: a real call. |
| 2 | SEI only | `filters.DateRangeFilter` → `filters._truncate` | `filters.py:470` in lambdas in the class body | **jedi wrong**: real calls. The harness attributes no caller to a lambda scope. |
| 3 | SEI only | `tests…F.__init__` → `filterset.BaseFilterSet.__init__` | `test_filtering.py:647` `super().__init__(…)`, `F(FilterSet)` | **jedi wrong**: `FilterSet` defines no `__init__`, so `super()` reaches `BaseFilterSet`'s. |
| 4 | jedi only | `test_filter_found_for_uuidfield` → `BaseFilterSet.filter_for_field` | `test_filterset.py:118` `FilterSet.filter_for_field(f, "uuid")` | jedi right: a classmethod inherited by a class named directly. |
| 5 | jedi only | `test_return_none` → `BaseFilterSet.filter_for_field` | `test_filterset.py:167` | jedi right (same pattern). |
| 6 | jedi only | `test_internal_type_is_manytomany` → `TaggableManager.get_internal_type` | `tests.py:748` `TaggableManager().get_internal_type()` | jedi right: a method on a constructor's result. |
| 7 | jedi only | `test_is_bound_and_not_valid` → `BaseFilterSet.is_valid` | `test_forms.py:253` `f = self.F(…)`; `f.is_valid()` | jedi right: the class comes from a class attribute. |
| 8 | jedi only | `test_filter_for_RANGE_lookup` → `BaseFilterSet.filter_for_lookup` | `test_filterset.py:299` `FilterSet.filter_for_lookup(…)` | jedi right (class named directly). |
| 9 | jedi only | `test_reverse_relationship_lookup_expr` → `BaseFilterSet.filter_for_field` | `test_filterset.py:372` | jedi right (class named directly). |
| 10 | jedi only | `test_modified_default_lookup` → `BaseFilterSet.filter_for_field` | `test_filterset.py:207` | jedi right (class named directly). |
| 11 | jedi only | `test_unknown_field_ignore_behavior` → `BaseFilterSet.handle_unrecognized_field` | `test_filterset.py:254` `self.FilterSet.handle_unrecognized_field(…)` | jedi right: a class held in an attribute. |
| 12 | jedi only | `test_reverse_o2o_relationship` → `BaseFilterSet.filter_for_field` | `test_filterset.py:327` | jedi right (class named directly). |
| 13 | jedi only | `test_field_with_choices` → `BaseFilterSet.filter_for_field` | `test_filterset.py:139` | jedi right (class named directly). |
| 14 | jedi only | `test_filterset_fields_no_queryset` → `DjangoFilterBackend.get_filterset_class` | `test_backends.py:127` `backend = DjangoFilterBackend()`, bound once | jedi right: the class is imported through `django_filters.rest_framework`'s `__init__`, a re-export SEI does not follow to `backends`. |
| 15 | jedi only | `CSVRangeFilterTests.test_filtering` → `BaseFilterSet.is_valid` | `test_filtering.py:1993` `f` assigned four times | jedi right: a local bound more than once, which SEI refuses to type. |
| 16 | jedi only | `test_filter_for_ISNULL_lookup` → `BaseFilterSet.filter_for_lookup` | `test_filterset.py:286` | jedi right (class named directly). |
| 17 | jedi only | `filters.Filter` → `filters.Filter.label` | `filters.py` `label = property(**label())` in the class body | jedi right: a class-body call to a function defined in that body. |
| 18 | jedi only | `test_number_filter_max_value_validation` → `BaseFilterSet.is_valid` | `test_forms.py:265` | jedi right (as #7). |
| 19 | jedi only | `BaseCSVFieldTests.test_validation_error` → `BaseCSVField.clean` | `test_fields.py:260` `field = self.DecimalCSVField(…)` | jedi right: `BaseCSVField` precedes the Django base in the MRO. |
| 20 | jedi only | `RenameAttributesBase.__setattr__` → `RenameAttributesBase.get_name` | `utils.py:102` `metacls.get_name(name)` | jedi right: a metaclass method's first parameter is named `metacls`, not `self`/`cls`. |

**The largest single cause of misses** is #4's pattern, a method called on a
class named directly, usually a classmethod the class inherits. It accounts
for 8 of the 17 sampled misses, and 26 of all 92. SEI does not resolve it yet.
