import pandas as pd
import psycopg2
from tqdm import tqdm

connection = psycopg2.connect(**params)


def check_database():
    with connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS toy_example;")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS toy_example.titanic_raw_data (
                PassengerId INTEGER NOT NULL,
                Survived BOOLEAN NOT NULL,
                Pclass INTEGER,
                Name TEXT,
                Sex TEXT,
                Age REAL,
                SibSp INTEGER,
                Parch INTEGER,
                Ticket TEXT,
                Fare REAL,
                Cabin TEXT,
                Embarked TEXT,
                PRIMARY KEY(PassengerId)
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS toy_example.survive (
            PassengerId INTEGER NOT NULL,
            Probability REAL,
            FOREIGN KEY (PassengerId) REFERENCES toy_example.titanic_raw_data(PassengerId)
            );"""
        )

        connection.commit()


def persist_raw_data(reader_iterator):
    with connection.cursor() as cursor:
        for _, line in tqdm(reader_iterator, desc="Titanic raw data"):
            (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked) = line
            print((PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked))

            cursor.execute(
                """
                    INSERT INTO toy_example.titanic_raw_data(
                    PassengerId,
                    Survived,
                    Pclass,
                    Name,
                    Sex,
                    Age,
                    SibSp,
                    Parch,
                    Ticket,
                    Fare,
                    Cabin,
                    Embarked)
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(PassengerId) DO NOTHING;
                """,
                (PassengerId, bool(Survived), Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked),
            )

    connection.commit()


if __name__ == "__main__":
    check_database()
    df = pd.read_csv("titanic_dataset.csv")
    persist_raw_data(df.iterrows())
