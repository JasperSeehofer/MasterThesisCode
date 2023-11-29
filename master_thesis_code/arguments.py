# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import os
import argparse
import logging
from typing import List


from master_thesis_code.exceptions import ArgumentsError

_LOGGER = logging.getLogger()
_VALID_LOG_LEVELS = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]

class Arguments:
    """this class includes the parsed input arguments."""

    def __init__(self, parsed_arguments: argparse.Namespace):
        self._parsed_arguments = parsed_arguments
        self._working_directory_replaced: bool = False
        self._log_level_replaced: bool = False
        self._working_directory: str = parsed_arguments.working_directory
        if not os.path.isdir(self._working_directory):
            self._working_directory_replaced = True
            self._working_directory = os.getcwd()

        if parsed_arguments.log_level.upper() in _VALID_LOG_LEVELS:
            self._log_level: int = getattr(logging, parsed_arguments.log_level.upper())
        else:
            self._log_level_replaced = True
            self._log_level = logging.INFO

    @property
    def working_directory(self) -> str:
        """Path to the working directory, where temporary files are stored, default is the current working directory."""
        return self._working_directory
    
    @property
    def log_level(self) -> int:
        """Log level of the stream and file logger, default is log level 'INFO'."""
        return self._log_level
    
    @property
    def use_gpu(self) -> bool:
        """Boolean that indicates whether to use gpu for waveform generation or not."""
        return self._parsed_arguments.use_gpu
    
    @property
    def simulation_steps(self) -> int:
        """Number of waveforms generated in the simulation."""
        return self._parsed_arguments.simulation_steps
    
    @staticmethod
    def create(sys_args: List[str] = sys.argv[1:]) -> Arguments:


        parsed_arguments = _parse_arguments(sys_args)
        return Arguments(parsed_arguments=parsed_arguments)
    
    def validate(self) -> None:
        """Validate the parsed arguments."""
        if self._working_directory_replaced is True:
            _LOGGER.warning(
                f"The path to the provided working directory does not exist. It is replaced by "
                f"{self._working_directory}."
            )
        if self._log_level_replaced is True:
            _LOGGER.warning(
                f"The provided log level is not valid. Valid values are: {', '.join(_VALID_LOG_LEVELS)}."
                f"The log level is set to {logging.getLevelName(self._log_level)}")
        if self._parsed_arguments.use_gpu is True:
            _LOGGER.info(
                f"GPU acceleration for waveform generation is activated"
            )
        try:
            self._simulation_steps = int(self._parsed_arguments.simulation_steps)
        except ValueError as original_error:
            raise ArgumentsError(
                f"{self._parsed_arguments.simulation_steps} could not be converted to integer."
                "Please provide an integer value as follows '--simulation_steps <int>'.") from original_error

def _parse_arguments(arguments: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "working_directory",
        help="Path to the working directory, where temporary files are stored.",
    )
    parser.add_argument(
        "--simulation_steps",
        help="Number of waveforms that are generated for data evaluation. (default is 0)",
        default=0,
        type=int
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Flag to indicate that GPU should be used for waveform generation.",
    )
    parser.add_argument(
        "--log_level",
        nargs="?",
        default="INFO",
        help="Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'). Default is 'INFO'."
    )
    parsed_arguments: argparse.Namespace = parser.parse_args(arguments)
    return parsed_arguments