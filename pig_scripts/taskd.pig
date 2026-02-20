-- load data
friends = LOAD 'data/friends.csv' USING PigStorage(',') AS (PersonID:int, FriendID:int, DateOfFriendship:chararray, Desc:chararray);
pages = LOAD 'data/pages.csv' USING PigStorage(',') AS (PersonID:int, Name:chararray, Nationality:chararray, CountryCode:int, Hobby:chararray);

-- left join for users w no friends
joined_data = JOIN pages BY PersonID LEFT OUTER, friends BY FriendID;
grouped_data = GROUP joined_data BY (pages::PersonID, pages::Name);

-- join data to get count
task_d = FOREACH grouped_data GENERATE group.pages::Name AS OwnerName, COUNT(joined_data.friends::PersonID) AS connected;

DUMP task_d;