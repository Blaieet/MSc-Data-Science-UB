db.createUser(
	{
		user:"dbUser",
		pwd:"abc16819154",
		roles: [
			{
				role:"readWrite",
				db:"TwitterDocker"
			}
		]
	}
)