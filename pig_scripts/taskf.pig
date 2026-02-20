-- load data
friends = LOAD 'data/friends.csv' USING PigStorage(',') AS (PersonID:int, FriendID:int, DateOfFriendship:chararray, Desc:chararray);
logs = LOAD 'data/access_logs.csv' USING PigStorage(',') AS (AccessID:int, ByWho:int, WhatPage:int, TypeOfAccess:chararray, AccessTime:chararray);
pages = LOAD 'data/pages.csv' USING PigStorage(',') AS (PersonID:int, Name:chararray, Nationality:chararray, CountryCode:int, Hobby:chararray);

-- join friends and logs on p1 and p2
joined_data = JOIN friends BY (PersonID, FriendID) LEFT OUTER, logs BY (ByWho, WhatPage);

-- filter to keep no access logs and get distinct list
no_access = FILTER joined_data BY logs::AccessID IS NULL;
p1_distinct = DISTINCT (FOREACH no_access GENERATE friends::PersonID AS p1_ID);

-- join with pages to get p1 names
result_join = JOIN p1_distinct BY p1_ID, pages BY PersonID;
task_f = FOREACH result_join GENERATE p1_distinct::p1_ID, pages::Name;

DUMP task_f;