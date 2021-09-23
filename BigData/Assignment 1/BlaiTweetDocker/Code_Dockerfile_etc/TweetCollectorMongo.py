import pymongo
from twython import TwythonStreamer

# My Twitter credentials
credentials = {}
credentials['CONSUMER_KEY'] = "6pgszXos5GnZTTryk65WUkhPR"
credentials['CONSUMER_SECRET'] = "9kdrilhesKNvuKeJchkLXOPNQoeCymMPI13mxiFbJ3qb5exSIS"
credentials['ACCESS_TOKEN'] = "1109844914731450369-r4ygUzhopAOqPSp0yBmtNNutZFISI3"
credentials['ACCESS_SECRET'] = "q04p5gZ0Jbmlfp52f3weW55eEYjrTxS4MPyOTvfEHCh8j"

# MongoDB Credentials
credentials['MONGOPASS'] = "abc16819154"
credentials['MONGOUSER'] = "dbUser"
credentials['MONGODB'] = "admin"

# Database and collection creation
def createDB(url):
    try:
        conn=pymongo.MongoClient(url)
    except pymongo.errors.ConnectionFailure as e:
            print ("Could not connect to MongoDB: %s" % e) 
    db = conn["Twitter"]
    return db['Tweets']

# URL for mongoDB Connection creation
def get_url():
    return "mongodb://"+credentials['MONGOUSER']+":"+credentials['MONGOPASS']+"@mongodb:27017/"+credentials["MONGODB"]+"?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&ssl=false"
    

# Filter out unwanted data
def process_tweet(tweet):
    d = {}
    d['hashtags'] = [hashtag['text'] for hashtag in tweet['entities']['hashtags']]
    d['user'] = tweet['user']
    d['created_at']=tweet['created_at']
    d['geo']=tweet['geo']
    d['reply_count']=tweet['reply_count']
    d['retweet_count']=tweet['retweet_count']
    d['favorite_count']=tweet['favorite_count']
    d['id']=tweet['id_str']
    d['in_reply_to_status_id']=tweet['in_reply_to_status_id_str']
    d['in_reply_to_user_id_str']=tweet['in_reply_to_user_id_str']
    return d


# Create a class that inherits TwythonStreamer
class MyStreamer(TwythonStreamer):
    
    collection = createDB(get_url())

    count=0
    # Received data
    def on_success(self, data):

        # Only collect tweets in English
        if data['lang'] == 'en':
            tweet_data = process_tweet(data)
            self.save_to_db(tweet_data)
            self.count+=1
            if self.count == 1:
                print("Connected succesfully on: ",get_url())
            if(self.count%10==0):
                print("Tweet received: "+str(self.count))
            

    # Problem with the API
    def on_error(self, status_code, data):
        print(status_code, data)
        self.disconnect()
        
    # Save each tweet to the MongoDB Collection
    def save_to_db(self, tweet):
        self.collection.insert_one(tweet)

# Instantiate from our streaming class
stream = MyStreamer(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'], 
                    credentials['ACCESS_TOKEN'], credentials['ACCESS_SECRET'])
# Start the stream
stream.statuses.filter(track='corona')
