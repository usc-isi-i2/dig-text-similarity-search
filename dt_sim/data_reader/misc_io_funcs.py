import os
import os.path as p
from pathlib import Path
from typing import Union


__all__ = ['check_unique', 'clear_dir']


def check_unique(test_path: Union[str, Path], count_mod: int = 0) -> str:
    """
    Checks for path uniqueness.
    Appends '_{count_mod}' to path (before file extension) if not unique.
    Useful for .index files, which will corrupt if overwritten.

    :param test_path: Path to test
    :param count_mod: Int to unique-ify test_path
    :return: Unique file path in target dir
    """
    if p.exists(test_path):
        print(f'\nWarning: File already exists  {test_path}')
        file_pth, file_ext = test_path.split('.')
        new_path = ''.join(file_pth.split('_')[:-1]) \
                   + f'_{count_mod}.{file_ext}'
        print(f'         Testing new path  {test_path}\n')
        count_mod += 1
        check_unique(new_path, count_mod=count_mod)
    else:
        new_path = test_path
    return new_path


def clear_dir(tmp_dir_path: Union[str, Path]):
    """
    Functionally equivalent to:
        $ rm -rf tmp_dir_path

    :param tmp_dir_path:
    """
    if not p.isdir(tmp_dir_path):
        print('Not a directory: {}'.format(tmp_dir_path))
    else:
        for (tmp_dir, _, tmp_files) in os.walk(tmp_dir_path):
            for file in tmp_files:
                os.remove(os.path.join(tmp_dir, file))
        os.rmdir(tmp_dir_path)
