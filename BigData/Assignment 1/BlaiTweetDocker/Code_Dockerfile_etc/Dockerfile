FROM python:3

 ADD TweetCollectorMongo.py /

 RUN pip install twython
 RUN pip install pymongo

 CMD [ "python", "./TweetCollectorMongo.py" ]