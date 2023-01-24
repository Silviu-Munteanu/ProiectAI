async function compare() {
  text1 = document.getElementById("text-1").value;
  text2 = document.getElementById("text-2").value;
  await fetch("/compare", {
    method: "POST",
    mode: "cors",
    body: JSON.stringify({
      text_1: text1,
      text_2: text2,
    }),
  });
}
