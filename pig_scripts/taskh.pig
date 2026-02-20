-- load data
friends = LOAD 'data/friends.csv' USING PigStorage(',') AS (PersonID:int, FriendID:int, DateOfFriendship:chararray, Desc:chararray);
pages = LOAD 'data/pages.csv' USING PigStorage(',') AS (PersonID:int, Name:chararray, Nationality:chararray, CountryCode:int, Hobby:chararray);

-- calculate friends per person
grouped_friends = GROUP friends BY FriendID;
friend_counts = FOREACH grouped_friends GENERATE group AS PersonID, COUNT(friends) AS num_friends;

-- group all and get avg count
all_counts = GROUP friend_counts ALL;
avg_calc = FOREACH all_counts GENERATE AVG(friend_counts.num_friends) AS overall_avg;

-- using CROSS (cartesian product) to friend count and avg
crossed = CROSS friend_counts, avg_calc;

-- only people greater than avg
popular_people = FILTER crossed BY friend_counts::num_friends > avg_calc::overall_avg;

-- join pages to get names
result_join = JOIN popular_people BY friend_counts::PersonID, pages BY PersonID;
task_h = FOREACH result_join GENERATE pages::Name, popular_people::friend_counts::num_friends;

DUMP task_h;