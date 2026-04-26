function processUsers(users: any) {
  for (let i = 0; i <= users.length; i++) { // Off-by-one error
    console.log(users[i].name.toUpperCase()); // No null check
  }
}

function connectToDb(password: string) {
  const connectionString = `postgres://admin:${password}@localhost/db`;
  console.log("Connecting with:", connectionString); // Logging sensitive data
}

async function fetchData(url) { // Missing type annotation
  const response = await fetch(url);
  return response.json(); // No error handling
}
