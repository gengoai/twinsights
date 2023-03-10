import json
from typing import Dict

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import sessionmaker


class JsonEncodedDict(sqlalchemy.types.TypeDecorator):
    impl = sqlalchemy.Text

    @property
    def python_type(self):
        return Dict

    def process_literal_param(self,
                              value,
                              dialect):
        return self.process_bind_param(value, dialect)

    def process_bind_param(self,
                           value,
                           dialect):
        if value is None:
            return value
        return json.dumps(value)

    def process_result_value(self,
                             value,
                             dialect):
        if value is None:
            return value
        return json.loads(value)


json_encoded_dict_type = MutableDict.as_mutable(JsonEncodedDict)


class DataStore:

    def __init__(self,
                 db_file: str,
                 base):
        self.engine = create_engine(f'sqlite:///{db_file}')
        base.metadata.create_all(self.engine)
        self.session: sqlalchemy.orm.Session = sessionmaker(bind=self.engine,
                                                            autoflush=False,
                                                            autocommit=False,
                                                            expire_on_commit=False)()

    def __enter__(self):
        return self

    def __exit__(self,
                 exc_type,
                 exc_val,
                 exc_tb):
        self.session.close()
