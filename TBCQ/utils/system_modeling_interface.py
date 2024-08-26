#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import sys
from TBCQ.utils import get_project_absoulte_path
import traceback


class SystemPathModifier:
    """
    """
    def __init__(self, add=None, delete=None) -> None:
        self.add = add
        self.delete = delete
        self.is_delete = False

    def __enter__(self):
        import sys
        sys.path.append(self.add)
        if self.delete in sys.path:
            sys.path.remove(self.delete)
            self.is_delete = True

    def __exit__(self, exc_type, exc_value, traceback):
        import sys
        sys.path.remove(self.add)
        if self.is_delete:
            sys.path.append(self.delete)


with SystemPathModifier(
        add=os.path.join(get_project_absoulte_path(), 'system_modeling'),
        delete=get_project_absoulte_path()
):
    try:
        from model_render import SystemModel
    except ImportError as e:
        from TBCQ.system_modeling.model_render import SystemModel
        traceback.print_exc()
