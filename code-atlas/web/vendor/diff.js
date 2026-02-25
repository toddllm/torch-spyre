(function () {
  function lineDiff(aText, bText) {
    var a = String(aText || "").split("\n");
    var b = String(bText || "").split("\n");
    var m = a.length;
    var n = b.length;
    var dp = Array(m + 1)
      .fill(0)
      .map(function () {
        return Array(n + 1).fill(0);
      });

    for (var i = m - 1; i >= 0; i--) {
      for (var j = n - 1; j >= 0; j--) {
        if (a[i] === b[j]) dp[i][j] = dp[i + 1][j + 1] + 1;
        else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }

    var out = [];
    var x = 0;
    var y = 0;
    while (x < m && y < n) {
      if (a[x] === b[y]) {
        out.push({ type: "same", text: a[x] });
        x++;
        y++;
      } else if (dp[x + 1][y] >= dp[x][y + 1]) {
        out.push({ type: "del", text: a[x] });
        x++;
      } else {
        out.push({ type: "add", text: b[y] });
        y++;
      }
    }
    while (x < m) {
      out.push({ type: "del", text: a[x++] });
    }
    while (y < n) {
      out.push({ type: "add", text: b[y++] });
    }
    return out;
  }

  window.DiffUtil = { lineDiff: lineDiff };
})();
