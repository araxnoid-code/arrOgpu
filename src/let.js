fetch("http://localhost:8000/")
  .then((res) => res.json())
  .then((res) => console.log(res))
  .catch(() => console.log("let it happen"));
