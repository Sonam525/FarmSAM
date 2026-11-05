"""tol_paths: stdlib-only loader of the catalog registry `_admin/registry/paths_v<N>.yaml`.

    import tol_paths
    R = tol_paths.load()                                   # env CALF_PATHS, else the highest paths_v<N>.yaml found
    R.path("paper2.outputs.masks_refined", day="20240424") # /Volumes/.../Tol/Tracking_v5a10_refined/20240424
    R.dbfs("datasets.tol_ch01.videos")                     # dbfs:/mnt/playbehavior/Tol/Videos
    R.out("paper1.bench", "pigtrack", "T")                 # a WRITE path: refused for Tol/, the mount and read-only keys
    R.days("test"); R.split_of("20240423")                 # [...], "val"
    R.gt("20240423")                                       # GT version selected in CURRENT.yaml (0423-TS -> two parts)
    R.gt("20240422", version="0422-S")                     # explicit version (Paper 1 calf windows pin their version)
    R.resolve_old("mount:Pig Behavior Edinburgh/annotated/2019_11_05/000002/output.json")
    R.provenance(["paper2.outputs.masks_refined"])         # embed in every output JSON

Rules enforced here: day ids match ^2024\\d{4}$ (guard H1); `external.*` keys and every Edinburgh mount key need
allow_external=True, and a mount path that has a catalog copy (registry `relocations`) is refused even then (D5: new
code reads Edinburgh from the catalog only); nothing returned by `out()` lies under `Tol/`, on the mount or under a
read-only key ("new code never writes into Tol/").
The YAML reader supports exactly the subset the registry uses: block mappings, "- " sequences of flow maps or
scalars, one-line flow maps / flow lists, quoted and plain scalars, comments. Run `python tol_paths.py --selftest`.
"""
import glob
import hashlib
import json
import os
import re
import sys

DAY_RX = re.compile(r"^2024\d{4}$")
EDINBURGH_MOUNT = "Pig Behavior Edinburgh"   # D5: every mount key under it needs allow_external=True
VERSION_RX = re.compile(r"^paths_v(\d+)\.yaml$")
DEFAULT_REGISTRY_DIR = "/Volumes/cornell_catalog/cornell_schema/blob_video/SAM3 Distillation and re-id/_admin/registry"


class RegistryError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------- YAML subset
def _strip_comment(line):
    out, q = [], None
    for i, ch in enumerate(line):
        if q:
            out.append(ch)
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
            out.append(ch)
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def _scalar(tok):
    t = tok.strip()
    if not t:
        return None
    if t[0] in "\"'":
        if len(t) < 2 or t[-1] != t[0]:
            raise RegistryError(f"unterminated string: {tok!r}")
        body = t[1:-1]
        return body.replace('\\"', '"').replace("\\\\", "\\") if t[0] == '"' else body.replace("''", "'")
    low = t.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~"):
        return None
    if re.match(r"^-?\d+$", t) and not (len(t) > 1 and t.lstrip("-").startswith("0")):
        return int(t)
    if re.match(r"^-?\d+\.\d+$", t):
        return float(t)
    return t


class _Flow:
    def __init__(self, s):
        self.s, self.i = s, 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def ch(self):
        if self.i >= len(self.s):
            raise RegistryError(f"unexpected end of flow value in {self.s!r}")
        return self.s[self.i]

    def value(self, stops):
        self.ws()
        if self.i >= len(self.s):
            raise RegistryError(f"unexpected end in {self.s!r}")
        ch = self.s[self.i]
        if ch == "{":
            return self.mapping()
        if ch == "[":
            return self.seq()
        if ch in "\"'":
            j = self.i + 1
            while j < len(self.s):
                if self.s[j] == ch and (ch == "'" or self.s[j - 1] != "\\"):
                    if ch == "'" and j + 1 < len(self.s) and self.s[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            tok = self.s[self.i:j + 1]
            self.i = j + 1
            return _scalar(tok)
        j = self.i
        while j < len(self.s) and self.s[j] not in stops:
            j += 1
        tok = self.s[self.i:j]
        self.i = j
        return _scalar(tok)

    def mapping(self):
        self.i += 1
        out = {}
        while True:
            self.ws()
            if self.ch() == "}":
                self.i += 1
                return out
            key = self.value(":")
            self.ws()
            if self.ch() != ":":
                raise RegistryError(f"expected ':' in {self.s!r}")
            self.i += 1
            out[str(key)] = self.value(",}")
            self.ws()
            if self.ch() == ",":
                self.i += 1
            elif self.ch() != "}":
                raise RegistryError(f"expected ',' or '}}' in {self.s!r}")

    def seq(self):
        self.i += 1
        out = []
        while True:
            self.ws()
            if self.ch() == "]":
                self.i += 1
                return out
            out.append(self.value(",]"))
            self.ws()
            if self.ch() == ",":
                self.i += 1
            elif self.ch() != "]":
                raise RegistryError(f"expected ',' or ']' in {self.s!r}")


def _inline(text):
    t = text.strip()
    if t.startswith("{") or t.startswith("["):
        f = _Flow(t)
        v = f.value("")
        f.ws()
        if f.i != len(t):
            raise RegistryError(f"trailing text after flow value: {t!r}")
        return v
    return _scalar(t)


def _split_key(line):
    """'key: value' -> (key, value-text). Keys may be quoted."""
    s = line.strip()
    if s[0] in "\"'":
        q = s[0]
        j = s.index(q, 1)
        key = s[1:j]
        rest = s[j + 1:].lstrip()
        if not rest.startswith(":"):
            raise RegistryError(f"expected ':' after key in {line!r}")
        return key, rest[1:].strip()
    m = re.match(r"^([^:]+?):(\s+(.*))?$", s)
    if not m:
        raise RegistryError(f"not a 'key: value' line: {line!r}")
    return m.group(1).strip(), (m.group(3) or "").strip()


def parse_yaml_subset(text):
    lines = []
    for raw in text.splitlines():
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise RegistryError("tabs are not allowed for indentation")
        s = _strip_comment(raw)
        if s.strip():
            lines.append((len(s) - len(s.lstrip(" ")), s.strip()))

    def block(i, indent):
        if i < len(lines) and lines[i][1].startswith("- "):
            out = []
            while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
                out.append(_inline(lines[i][1][2:]))
                i += 1
            return out, i
        out = {}
        while i < len(lines) and lines[i][0] == indent:
            key, val = _split_key(lines[i][1])
            if key in out:
                raise RegistryError(f"duplicate key {key!r}")
            i += 1
            if val == "":
                if i < len(lines) and lines[i][0] > indent:
                    out[key], i = block(i, lines[i][0])
                else:
                    out[key] = None
            else:
                out[key] = _inline(val)
        return out, i

    if not lines:
        return {}
    data, i = block(0, lines[0][0])
    if i != len(lines):
        raise RegistryError(f"bad indentation near: {lines[i][1]!r}")
    return data


# ---------------------------------------------------------------------------------------------------- registry
class Registry:
    def __init__(self, data, source="", sha256="", current_override=None):
        self.data = data
        self.source = source
        self.sha256 = sha256
        self._current_override = current_override
        roots = data.get("roots") or {}
        self.volume = str(roots.get("volume", "")).rstrip("/")
        self.mount = str(roots.get("mount", "/dbfs/mnt/playbehavior")).rstrip("/")
        self.mount_uri = str(roots.get("mount_uri", "dbfs:/mnt/playbehavior")).rstrip("/")
        if not self.volume:
            raise RegistryError("registry lacks roots.volume")
        self._check()

    # -- structure
    def _node(self, key):
        node = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise RegistryError(f"unknown registry key {key!r}")
            node = node[part]
        return node

    def leaf_keys(self):
        out = []

        def rec(prefix, node):
            if isinstance(node, dict) and "space" in node and "path" in node:
                out.append(prefix)
            elif isinstance(node, dict):
                for k, v in node.items():
                    rec(f"{prefix}.{k}" if prefix else k, v)
        for top in ("datasets", "models", "paper1", "paper2", "releases", "legacy", "external", "admin"):
            if top in self.data:
                rec(top, self.data[top])
        return out

    def _check(self):
        days = self.data.get("days") or {}
        for split in ("train", "val", "test"):
            for d in days.get(split) or []:
                if not DAY_RX.match(str(d)):
                    raise RegistryError(f"bad day id {d!r} in days.{split}")
        alld = [d for s in ("train", "val", "test") for d in days.get(s) or []]
        if len(alld) != len(set(alld)):
            raise RegistryError("a day appears in two splits")
        for k in self.leaf_keys():
            n = self._node(k)
            if n["space"] not in ("volume", "mount"):
                raise RegistryError(f"{k}: space must be volume|mount")
            if str(n["path"]).startswith("/") or ".." in str(n["path"]).split("/"):
                raise RegistryError(f"{k}: path must be relative to its root without '..'")

    # -- days
    @staticmethod
    def check_day(day):
        if not DAY_RX.match(str(day)):
            raise RegistryError(f"day must look like 20240424, got {day!r}")
        return str(day)

    def days(self, split):
        if split == "validation":
            split = "val"
        d = (self.data.get("days") or {}).get(split)
        if d is None:
            raise RegistryError(f"unknown split {split!r} (train | val | test)")
        return list(d)

    def split_of(self, day):
        day = self.check_day(day)
        for split in ("train", "val", "test"):
            if day in self.days(split):
                return split
        raise RegistryError(f"day {day} is in no split")

    # -- paths
    def entry(self, key, allow_external=False):
        if key.startswith("external.") and not allow_external:
            raise RegistryError(f"{key}: external (other projects); pass allow_external=True to read it")
        n = self._node(key)
        if not (isinstance(n, dict) and "space" in n and "path" in n):
            raise RegistryError(f"{key} is not a path entry")
        if (n["space"] == "mount" and str(n["path"]).startswith(EDINBURGH_MOUNT) and not allow_external):
            raise RegistryError(f"{key}: Edinburgh on the blob mount; new code reads Edinburgh from the catalog only (D5). "
                                "Pass allow_external=True for a deliberate legacy read")
        return n

    def _join(self, n, day=None, extra=()):
        base = (self.volume if n["space"] == "volume" else self.mount) + "/" + str(n["path"]).strip("/")
        parts = []
        if day is not None:
            parts.append(self.check_day(day))
        elif n.get("per_day") and not extra:
            pass
        parts += [str(e).strip("/") for e in extra if str(e).strip("/")]
        for p in parts:
            if ".." in p.split("/"):
                raise RegistryError("'..' not allowed")
        out = "/".join([base] + parts)
        if n["space"] == "mount":
            rel = out[len(self.mount) + 1:]
            moved = self.resolve_old("mount:" + rel)
            if moved is not None:
                raise RegistryError(f"mount:{rel} has a catalog copy ({moved['new']}, op {moved['op']}); read the catalog (D5)")
        return out

    def path(self, key, day=None, *extra, allow_external=False):
        n = self.entry(key, allow_external=allow_external)
        if day is not None and not n.get("per_day") and not extra:
            raise RegistryError(f"{key} is not a per-day entry")
        return self._join(n, day, extra)

    def dbfs(self, key, day=None, *extra, allow_external=False):
        p = self.path(key, day, *extra, allow_external=allow_external)
        if p.startswith(self.mount + "/") or p == self.mount:
            return self.mount_uri + p[len(self.mount):]
        return "dbfs:" + p

    def out(self, key, *extra, day=None):
        """A path new code may WRITE. Refuses read-only keys, the mount, and anything under Tol/."""
        n = self.entry(key)
        if n.get("read_only"):
            raise RegistryError(f"{key} is read-only")
        if n["space"] != "volume":
            raise RegistryError(f"{key}: writing to the mount is not allowed")
        p = self._join(n, day, extra)
        if p == self.volume + "/Tol" or p.startswith(self.volume + "/Tol/"):
            raise RegistryError(f"{key}: new code never writes into Tol/")
        return p

    # -- ground truth
    def _current(self, current_path=None):
        if self._current_override is not None:
            return self._current_override
        p = current_path or self.path("datasets.tol_ch01.ground_truth_current")
        with open(p, encoding="utf-8") as f:
            text = f.read()
        return {"data": parse_yaml_subset(text), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "path": p}

    def gt(self, day, version=None, current_path=None):
        """{'day','version','parts':[{'version','path','window'}], 'current_sha256'}; composite versions give 2 parts."""
        day = self.check_day(day)
        cur = None
        if version is None:
            cur = self._current(current_path)
            sel = ((cur["data"].get("days") or {}).get(day) or {}).get("selected")
            if not sel:
                raise RegistryError(f"CURRENT.yaml selects no GT version for {day}")
            version = sel
        versions = self.data.get("gt_versions") or {}
        if version not in versions:
            raise RegistryError(f"unknown GT version {version!r}")
        v = versions[version]
        if day not in [str(d) for d in v.get("days", [])]:
            raise RegistryError(f"GT version {version} does not cover {day}")
        root = self.path("datasets.tol_ch01.ground_truth")
        parts = []
        if v.get("composite"):
            for sub in v["composite"]:
                sv = versions[sub]
                parts.append({"version": sub, "path": root + "/" + sv["path"], "window": sv.get("window", "")})
        else:
            parts.append({"version": version, "path": root + "/" + v["path"], "window": v.get("window", "")})
        return {"day": day, "version": version, "parts": parts, "rule": v.get("rule", ""),
                "current_sha256": cur["sha256"] if cur else None}

    # -- old paths
    def resolve_old(self, old):
        """Longest-prefix match of 'space:path' in relocations -> (new 'space:path', relation, op, verified) or None."""
        best = None
        for r in self.data.get("relocations") or []:
            o = r["old"].rstrip("/")
            if old == o or old.startswith(o + "/"):
                if best is None or len(o) > len(best["old"].rstrip("/")):
                    best = r
        if best is None:
            return None
        tail = old[len(best["old"].rstrip("/")):]
        return {"new": best["new"].rstrip("/") + tail, "relation": best["relation"], "op": best["op"],
                "verified": bool(best.get("verified"))}

    def provenance(self, keys, day=None, gt_day=None):
        out = {"registry_file": self.source, "registry_sha256": self.sha256, "version": self.data.get("version"),
               "resolved": {}}
        for k in keys:
            n = self.entry(k, allow_external=True)          # recording a path is not reading it
            out["resolved"][k] = self._join(n, day if n.get("per_day") else None, ())
        if gt_day is not None:
            g = self.gt(gt_day)
            out["gt"] = {"day": g["day"], "version": g["version"], "current_sha256": g["current_sha256"]}
        return out


def load(path=None, registry_dir=None, version=None, current_override=None):
    """Load a registry file: explicit path, else env CALF_PATHS, else the highest (or the pinned) version in a folder."""
    path = path or os.environ.get("CALF_PATHS")
    if not path:
        d = registry_dir or os.environ.get("CALF_PATHS_DIR") or DEFAULT_REGISTRY_DIR
        found = {}
        for p in glob.glob(os.path.join(d, "paths_v*.yaml")):
            m = VERSION_RX.match(os.path.basename(p))
            if m:
                found[int(m.group(1))] = p
        if not found:
            raise RegistryError(f"no paths_v<N>.yaml in {d}")
        path = found[version] if version is not None else found[max(found)]
    with open(path, "rb") as f:
        raw = f.read()
    data = parse_yaml_subset(raw.decode("utf-8"))
    return Registry(data, source=path, sha256=hashlib.sha256(raw).hexdigest(), current_override=current_override)


# ---------------------------------------------------------------------------------------------------- self-test
def selftest(registry_path=None, current_path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    registry_path = registry_path or os.path.join(here, "paths_v1.yaml")
    current_path = current_path or os.path.join(here, "plan_docs", "datasets__calves_tol_ch01__ground_truth__CURRENT.yaml")
    res = []

    def check(name, cond, detail=""):
        res.append((name, bool(cond)))
        print(("PASS " if cond else "FAIL ") + name + ("" if cond or not detail else f"  [{detail}]"))

    def raises(fn):
        try:
            fn()
            return False
        except RegistryError:
            return True

    # YAML subset unit tests
    y = parse_yaml_subset('a: 1\nb: "x # not a comment"  # comment\nc:\n  d: [1, "two", {e: f}]\n  "2024": {k: v, l: [a, b]}\n'
                          "s:\n  - {old: \"x:y\", n: 2}\n  - plain\nt: 'it''s'\nu: true\nv: 0422-S\nw:\n")
    check("yaml: scalars, comments, quotes", y["a"] == 1 and y["b"] == "x # not a comment" and y["t"] == "it's" and y["u"] is True
          and y["v"] == "0422-S" and y["w"] is None, str(y))
    check("yaml: nested flow + quoted keys", y["c"]["d"] == [1, "two", {"e": "f"}] and y["c"]["2024"] == {"k": "v", "l": ["a", "b"]})
    check("yaml: block sequence of flow maps", y["s"] == [{"old": "x:y", "n": 2}, "plain"])
    check("yaml: duplicate key refused", raises(lambda: parse_yaml_subset("a: 1\na: 2\n")))
    check("yaml: tab indentation refused", raises(lambda: parse_yaml_subset("a:\n\tb: 1\n")))
    check("yaml: unterminated flow refused", raises(lambda: parse_yaml_subset("a: {b: 1\n")))

    R = load(registry_path)
    check("registry loads, version 1", R.data.get("version") == 1, str(R.data.get("version")))
    try:
        import yaml  # optional cross-check, not a dependency
        with open(registry_path, encoding="utf-8") as f:
            ref = yaml.safe_load(f)
        # PyYAML reads unquoted 0422-S etc. as strings too; compare after json round-trip of str keys
        norm = lambda o: json.loads(json.dumps(o, default=str, sort_keys=True))
        check("registry parses identically with PyYAML", norm(ref) == norm(R.data))
    except ImportError:
        print("SKIP PyYAML cross-check (not installed)")
    check("days: train", R.days("train") == ["20240418", "20240419"])
    check("days: val", R.days("val") == ["20240422", "20240423"] and R.days("validation") == R.days("val"))
    check("days: test (04-24 is test, D1)", R.days("test") == ["20240424", "20240425", "20240426", "20240427", "20240428"])
    check("split_of", R.split_of("20240419") == "train" and R.split_of("20240423") == "val" and R.split_of("20240424") == "test")
    check("bad day refused (H1)", raises(lambda: R.split_of("2024-04-24")) and raises(lambda: R.path("paper2.outputs.masks_refined", "0424")))
    vol = R.volume
    check("per-day path", R.path("paper2.outputs.masks_refined", "20240424") == vol + "/Tol/Tracking_v5a10_refined/20240424")
    check("non-per-day key refuses a day", raises(lambda: R.path("models.student_tol_v2", "20240424")))
    check("dbfs form for the mount", R.dbfs("datasets.tol_ch01.videos") == "dbfs:/mnt/playbehavior/Tol/Videos")
    check("dbfs form for the volume", R.dbfs("models.student_tol_v2").startswith("dbfs:/Volumes/"))
    check("external refused without allow_external", raises(lambda: R.path("external.behaviour_2025")))
    check("external allowed with allow_external", R.path("external.behaviour_2025", allow_external=True) == "/dbfs/mnt/playbehavior/Tol")
    check("D5: legacy Edinburgh mount key refused without allow_external", raises(lambda: R.path("legacy.edinburgh_teacher_masks")))
    check("D5: legacy Edinburgh mount key allowed with allow_external",
          R.path("legacy.edinburgh_teacher_masks", allow_external=True).endswith("Pig Behavior Edinburgh/Model_Distill/SAM3/Masks"))
    check("D5: a copied Edinburgh folder is refused on the mount even with allow_external",
          raises(lambda: R.path("external.edinburgh_2025", None, "annotated", "2019_11_05", allow_external=True))
          and raises(lambda: R.path("external.edinburgh_2025", None, "Decoded Frames", allow_external=True))
          and raises(lambda: R.path("external.edinburgh_2025", None, "Model_Distill/SAM3/Checkpoints_Backbone/best_backbone_student.pt", allow_external=True)))
    check("D5: an uncopied 2025 Edinburgh folder still resolves with allow_external",
          R.path("external.edinburgh_2025", None, "YOLO", allow_external=True).endswith("Pig Behavior Edinburgh/YOLO"))
    check("D11: no Tol2 key anywhere in the registry", "Tol2" not in json.dumps(R.data) and "tol2" not in json.dumps(R.data))
    check("out(): refuses read-only Tol roots", raises(lambda: R.out("paper2.outputs.masks_refined")))
    check("out(): refuses the mount", raises(lambda: R.out("datasets.tol_ch01.videos")))
    check("out(): allows new homes", R.out("paper1.bench", "pigtrack", "T") == vol + "/paper1_distillation/bench/pigtrack/T")
    check("out(): refuses '..'", raises(lambda: R.out("paper1.bench", "../Tol/x")))
    keys = R.leaf_keys()
    check("every leaf key resolves", all(R.path(k, allow_external=True) for k in keys), str(len(keys)))
    check("no Edinburgh mount key outside legacy/external (D5)",
          not [k for k in keys if not k.startswith(("legacy.", "external.")) and "Pig Behavior Edinburgh" in str(R._node(k)["path"])])
    check("no Tol2 key (D11)", not [k for k in keys if "Tol2" in str(R._node(k)["path"])])
    check("every writable volume key lies outside Tol/",
          all(not str(R._node(k)["path"]).startswith("Tol") for k in keys if R._node(k)["space"] == "volume" and not R._node(k).get("read_only")))
    check("GT root is under the calf dataset (D2)", R.path("datasets.tol_ch01.ground_truth").endswith("datasets/calves_tol_ch01/ground_truth/v1"))
    if os.path.isfile(current_path):
        g = R.gt("20240423", current_path=current_path)
        check("gt(): CURRENT selects 0423-TS = teacher evening + student chain", g["version"] == "0423-TS"
              and [p["version"] for p in g["parts"]] == ["0423-T", "0423-S"] and g["current_sha256"])
        check("gt(): 04-18 -> TRN-T", R.gt("20240418", current_path=current_path)["version"] == "TRN-T")
        check("gt(): 04-24 -> 0424-S", R.gt("20240424", current_path=current_path)["parts"][0]["path"].endswith("test_20240424/0424-S_student_v4v2p2"))
        check("gt(): explicit pin 0422-S", R.gt("20240422", version="0422-S")["parts"][0]["path"].endswith("20240422/0422-S_student_v2_chain"))
        check("gt(): version/day mismatch refused", raises(lambda: R.gt("20240424", version="0422-S")))
        prov = R.provenance(["paper2.outputs.masks_refined"], day="20240424", gt_day=None)
        check("provenance carries the registry sha", prov["registry_sha256"] == R.sha256 and prov["resolved"]["paper2.outputs.masks_refined"].endswith("/20240424"))
    else:
        print("SKIP gt(): no CURRENT.yaml at", current_path)
    r = R.resolve_old("mount:Pig Behavior Edinburgh/annotated/2019_11_05/000002/output.json")
    check("resolve_old: longest prefix inside a copied tree", r and r["new"] == "volume:datasets/edinburgh_pigs/annotated/2019_11_05/000002/output.json" and r["op"] == "E01", str(r))
    r = R.resolve_old("volume:Tol/Checkpoints/best_backbone_student_v2.pt")
    check("resolve_old: model file", r and r["new"].endswith("models/sam3_student_tinyvit/tol_ch01_v2/best_backbone_student_v2.pt"), str(r))
    check("resolve_old: unknown path -> None", R.resolve_old("volume:Tol/Tracking_v5a10/20240424/x.json") is None)
    n_fail = sum(1 for _, ok in res if not ok)
    print(f"\ntol_paths selftest: {len(res) - n_fail}/{len(res)} passed")
    return n_fail


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--selftest"]
        sys.exit(1 if selftest(*args[:2]) else 0)
    print(__doc__)
