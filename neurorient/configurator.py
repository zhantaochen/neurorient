from contextlib import contextmanager


class Configurator(dict):
    """
    Poorman's configurator, basically like AttrDict.

    Reference: https://github.com/facebookresearch/Detectron/blob/main/detectron/utils/collections.py
    """
    __auto_create = False

    def __getattr__(self, attr):
        if attr not in self:
            if self.__class__.__auto_create:
                self[attr] = Configurator()
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{attr}'. "
                    "If you're trying to auto-create attributes, consider using the 'enable_auto_create' context manager in a with-statement."
                )
        return self[attr]


    def __setattr__(self, attr, value):
        if attr == "_Configurator__auto_create":  # Use name mangling for private attributes
            raise AttributeError("'_auto_create' is read-only!")
        self[attr] = value    # Using __setitem__ under the hood


    @contextmanager
    def enable_auto_create(self):
        original_state = self.__class__.__auto_create
        self.__class__.__auto_create = True
        try:
            yield self
        finally:
            self.__class__.__auto_create = original_state


    def to_dict(self):
        result = {}
        for key, value in self.items():
            if isinstance(value, Configurator):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


    @classmethod
    def from_dict(cls, data):
        instance = cls()
        for key, value in data.items():
            if isinstance(value, dict):
                instance[key] = cls.from_dict(value)
            else:
                instance[key] = value

        return instance


    def get_value(self, key):
        """
        Retrieve a nested key using dot-separated notation.
        For example, for a key like "a.b.c", it fetches self['a']['b']['c'] if it exists, otherwise returns default.
        """
        keys  = key.split('.')
        value = self
        for k in keys:
            if k in value:
                value = value[k]
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{k}'. "
                )

        return value


    def set_value(self, key, value):
        """
        Set a nested key using dot-separated notation.
        For example, for a key like "a.b.c", it sets value to self['a']['b']['c'].
        Creates necessary nested Configurators if they don't exist.
        """
        keys = key.split('.')
        current_value = self
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                current_value[k] = value
            else:
                if k not in current_value or not isinstance(current_value[k], Configurator):
                    if self.__class__.__auto_create:
                        current_value[k] = Configurator()
                    else:
                        raise AttributeError(
                            f"'{type(self).__name__}' object has no attribute '{k}'. "
                            "If you're trying to auto-create attributes, consider using the 'enable_auto_create' context manager in a with-statement."
                        )
                current_value = current_value[k]

    def deep_merge(self, d_low_priority, d_high_priority):
        """
        Merges d2 into d1 recursively, updating keys with values from d2.
        """
        for k, v in d_high_priority.items():
            if isinstance(v, dict):
                d_low_priority[k] = self.deep_merge(d_low_priority.get(k, {}), v)
            else:
                d_low_priority[k] = v
        return d_low_priority

    def merge_with_priority(self, other, self_has_priority=True):
        if self_has_priority:
            low_priority_dict = other.to_dict()
            high_priority_dict = self.to_dict()
        else:
            low_priority_dict = self.to_dict()
            high_priority_dict = other.to_dict()
        out_dict = self.deep_merge(low_priority_dict, high_priority_dict)
        return Configurator.from_dict(out_dict)

    def dump_to_file(self, filepath):
        """Dump the contents of the configurator to a text file."""
        with open(filepath, 'w') as f:
            self._write_dict_content(self, f, 0, True)

    def _write_dict_content(self, current_dict, file_handle, indent_level, is_top_level=False):
        """Recursive helper function to write the contents of the dictionary to a file."""
        indent = '    '  # 2 spaces for indentation

        # Determine the maximum key length for alignment
        max_key_len = max([len(key) for key in current_dict.keys()])

        first_entry = True
        for key, value in current_dict.items():
            # For top-level keys, add an empty line for clarity, but skip the very first one
            if is_top_level and not first_entry:
                file_handle.write("\n")
            first_entry = False

            if isinstance(value, Configurator):
                file_handle.write(f"{indent * indent_level}{key.upper()}:\n")
                self._write_dict_content(value, file_handle, indent_level + 1)
            else:
                if value is None:
                    formatted_value = "null"
                elif isinstance(value, bool):
                    formatted_value = str(value).lower()
                elif isinstance(value, str):
                    formatted_value = f"'{value}'"
                else:
                    formatted_value = str(value)

                # Writing the key-value pair with alignment
                file_handle.write(f"{indent * indent_level}{key.upper().ljust(max_key_len)} : {formatted_value}\n")