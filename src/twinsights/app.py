import argparse
import inspect
import itertools
import logging
import typing
from collections import UserDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, \
    Type, \
    Union

from dotenv import find_dotenv, load_dotenv

COMMAND_NAME = "app_command_name"
COMMAND_ARGS = "app_command_args"
COMMAND_HELP = "app_command_help"


class NoCommandException(Exception):
    """
    Custom exception when no command is defined for an application.
    """

    def __init__(self):
        super(NoCommandException, self).__init__("No Commands Defined")


@dataclass
class Arg:
    """
    Dataclass wrapping the parameters for the argparse add_argument method.
    An Arg is used as part of the args parameter for the
    :func:`~gengoai.main.Command` decorate to specify a command line argument
    for one the decorated function's parameters.
    """
    #: The name of the function's argument parameter this Arg is bound to
    dest: str
    #: A list of the command line argument names (by default an empty list,
    # which will automatically create
    #: a command line argument of --dest where dest is the function's
    # argument name.
    option_names: List[str] = field(default_factory=list)
    #: The number of values that can be specified for this argument
    nargs: str = "?"
    #: The help message describing what the argument is used for
    help: str = ""
    #: The type of the argument either as a python type or a callable which
    # converts a string to the desired value
    type: Union[Type, Callable[[str], Any]] = None
    #: The default value of the argument
    default: Any = None
    #: Determines if it is required to specify the argument
    required: bool = False
    #: The possible values the argument can have
    choices: Optional[List[Any]] = None
    #: The action performed when a value is given for the argument.
    action: str = "store"

    def update(self,
               other: 'Arg') -> None:
        """
        Updates this arg with values from the given other arg using the
        following set of rules:

        +---------------+-------+--------+-----------+
        |     field     |  self |  other | selection |
        +---------------+-------+--------+-----------+
        | option_names  | empty |  any   |   other   |
        +---------------+-------+--------+-----------+
        | required      | any   |  True  |   other   |
        +---------------+-------+--------+-----------+
        | default       | None  |  any   |   other   |
        +---------------+-------+--------+-----------+
        | type          | str   |  any   |   other   |
        +---------------+-------+--------+-----------+
        | choices       | empty |  any   |   other   |
        +---------------+-------+--------+-----------+
        | help          | empty |  any   |   other   |
        +---------------+-------+--------+-----------+

        :param other: the Arg class we will update with
        :type other: Arg
        """
        if other is None:
            return
        if len(self.option_names) == 0:
            self.option_names = other.option_names
        if other.required:
            self.required = True
        if self.default is None:
            self.default = other.default
        if self.type is None or self.type == str:
            self.type = other.type
        if self.choices is None or len(self.choices) == 0:
            self.choices = other.choices
        if self.help == "":
            self.help = other.help


@dataclass
class Command:
    """
    Decorator for defining a method as a Command.
    """
    name: Optional[str] = None
    help: str = ""
    args: Iterable[Arg] = field(default_factory=list)

    def __call__(self,
                 func):
        setattr(func, COMMAND_NAME,
                func.__name__ if self.name is None else self.name)
        setattr(func, COMMAND_ARGS,
                CommandArgs(inspect.signature(func).parameters, self.args))
        setattr(func, COMMAND_HELP, self.help)
        return func


class CommandArgs(UserDict):
    """
    Customized dictionary of argument name to **Arg**.
    """

    def __init__(self,
                 func_parameters,
                 arg_annotations):
        """

        :param func_parameters:
        :type func_parameters:
        :param arg_annotations:
        :type arg_annotations:
        """
        super().__init__()
        self.update({param_name: CommandArgs._func_parameter_to_arg(
            func_parameters[param_name]) for
            param_name in list(func_parameters)[1:]})
        self.update({arg.dest: arg for arg in arg_annotations})

    def update(self,
               __m: Mapping[str, Arg],
               **kwargs) -> None:
        """
        Updates this CommandArgs with the values in the given mapping.

        :param __m: The mapping to update from
        :type __m:   Mapping[str, Arg]
        :param kwargs: Any other args
        """
        for name, arg in __m.items():
            arg.update(self.get(arg.dest, None))
            self[name] = arg

    @staticmethod
    def _func_parameter_to_arg(parameter):
        arg_class = parameter.annotation
        default_value = parameter.default
        is_required = False
        action = "store"
        if hasattr(arg_class, '__name__') and arg_class.__name__ == '_empty':
            arg_class = str

        if isinstance(default_value,
                      type) and default_value.__name__ == '_empty':
            default_value = None
            is_required = True
        elif arg_class == bool:
            action = "store_true" if default_value is False else "store_false"
        return Arg(parameter.name,
                   option_names=[f"--{parameter.name}"],
                   type=arg_class,
                   default=default_value,
                   action=action,
                   required=is_required)


class App:
    """
    Object that defines an application as one more commands that be executed.
    To create an application you must create a child class of App and add a
    method for each  command using the :class:`~Command` decorator. The
    command decorator can include a *name* for the command, *help* for the
    command, and a list of :class:`~Arg` which define the command line
    arguments. Note that App class is smart enough to infer basic arguments
    using the parameter names, type hints, and default values in the function
    signature. For finer grained control the **Arg** class can be used,which is
    a basic wrapper for argparse. An example application is as follows:

    .. sourcecode:: pycon

        >>> class MyApp(App)
        >>>
        >>>     @Command(help="Processes the given file and outputs the results"
        >>>                    " to the terminal",
        >>>              args=[Arg(dest="filename",
        >>>                        help="The filename to process")]
        >>>     def process(self, filename:str):
        >>>         # do some logic
        >>>        print()
        >>>
        >>>
        >>>     MyApp().run() # Run the app

    In the example, shown above, we define process as a command, we do not
    specify a name so by default the function name, 'process', is used. We
    define a description that will be used when displaying help. We also define
    the single argument parameter. We will let the 'filename' argument get its
    type from the function signature and also that it is required (as there is
    no default value.) The Arg class is only defining a help message to display.
    The last line of the example code creates an instance of the application and
    runs it. Note that since only one command is defined, running the
    application will directly run the single command and will have a command
    line as follows:

    .. code-block:: console

        python app.py --filename=<FILENAME>

    If we had defined a second command named *print*, then the command line
    would be as follows:

    .. code-block:: console

        python app.py  [{process, print}]

    where the first argument is the command to run. The app will then create
    a new parser to parse the command specific
    arguments.

    All Apps, will have the following command line arguments::

        -v, --version # prints the version number of the application
        -h, --help    # displays help for the command or application
        --log-level   # defines the default logging level
        --log-file    # if specified, will output a log file in the working
        directory named *MyApp.log*

    App defines two specified fields ::

        __description__
        __version__

    That defines a helpful description of the app and the current version
    number of the app.
    """
    __description__: str = ""
    __version__: str = "0.1"

    def __init__(self):
        load_dotenv(find_dotenv())
        self.logger: Optional[logging.Logger] = None
        self._commands: Dict[str, Callable] = {}
        self._global_arguments: List[Arg] = [
            Arg(dest='log_file',
                option_names=['--log-file'],
                action='store_true',
                help='Save the log to a file named App.log',
                type=bool,
                default=False),
            Arg(dest='log_level',
                option_names=['--log-level'],
                action='store',
                help='The level of log messages to display',
                choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                default='INFO')
        ]
        self.parsed_global_args: Dict[str, Any] = dict()
        for cmd_name, cmd_ptr in inspect.getmembers(self,
                                                    predicate=inspect.ismethod):
            if hasattr(cmd_ptr, COMMAND_NAME):
                self._commands[getattr(cmd_ptr, COMMAND_NAME)] = cmd_ptr

    def add_global_argument(self,
                            args: Union[Arg, List[Arg]]) -> None:
        """
        Adds one or more arguments as global, meaning they will be a command
        line argument for each command.

        Args:
            args (Union[Arg, List[Arg]): The global arguments
        """
        if isinstance(args, list):
            self._global_arguments.extend(args)
        else:
            self._global_arguments.append(args)

    def setup(self) -> None:
        """
        Runs specialized setup code before executing the command. By default
        this includes scanning the `gengoai` package and all of its subpackages
        to load required objects.
        """
        pass

    def run(self) -> None:
        """
        Runs the application, constructing the required argparser, and calls
        method for the user given command with
        user given arguments.
        """
        if len(self._commands) == 0:
            raise NoCommandException()
        elif len(self._commands) == 1:
            self._dispatch(next(iter(self._commands.values())))
        else:
            parser = argparse.ArgumentParser(description=self.__description__)
            parser.add_argument("option", choices=self._commands.keys(),
                                help="Which command to run")
            parser.add_argument("-v", "--version",
                                action="version",
                                version=f"{type(self).__name__} "
                                        f"{self.__version__}")
            args, unknown = parser.parse_known_args()
            self._dispatch(self._commands[args.option], unknown)

    def _dispatch(self,
                  cmd_func: Callable,
                  arguments=None):
        description = f"{self.__description__}\n{'-' * 20}\n" \
                      f"{getattr(cmd_func, COMMAND_HELP)}"

        parser = self._create_argparser(cmd_func, description)
        if arguments is None:
            parser.add_argument("-v", "--version",
                                action="version",
                                version=f"{type(self).__name__} "
                                        f"{self.__version__}")
        args = vars(parser.parse_args(arguments))

        log_formatter = logging.Formatter(
            '%(asctime)s %(levelname)s [%(name)s] %(message)s')
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.getLevelName(args['log_level']))

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        root_logger.handlers = []
        root_logger.addHandler(console_handler)

        if args['log_file']:
            file_handler = logging.FileHandler(f"./{type(self).__name__}.log",
                                               "w+")
            file_handler.setFormatter(log_formatter)
            root_logger.addHandler(file_handler)

        for arg in self._global_arguments:
            self.parsed_global_args[arg.dest] = args[arg.dest]
            del args[arg.dest]

        self.logger = logging.getLogger(type(self).__name__)
        self.setup()
        cmd_func(**args)

    def _create_argparser(self,
                          func,
                          description):
        parser = argparse.ArgumentParser(description=description)
        for arg in itertools.chain(iter(getattr(func, COMMAND_ARGS).values()),
                                   iter(self._global_arguments)):
            arg_dict = arg.__dict__
            names_or_flags = arg_dict['option_names']
            del arg_dict['option_names']
            if arg_dict['action'] != 'store':
                del arg_dict['nargs']
                del arg_dict['type']
                del arg_dict['choices']
            parser.add_argument(*names_or_flags, **arg_dict)

        return parser
