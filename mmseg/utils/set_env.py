import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:


    import mmseg.datasets
    import mmseg.engine
    import mmseg.evaluation
    import mmseg.models
    import mmseg.structures

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmseg')
        if never_created:
            DefaultScope.get_instance('mmseg', scope_name='mmseg')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmseg':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmseg", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmseg". If this is not '
                          'expected, please set `init_default_scope=False`.')

            new_instance_name = f'mmseg-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmseg')
