# SkinBouncer Paper plugin

Checks every joining player's skin against a running [SkinBouncer API](../06_Deployment/)
and warns staff when a detector flags it.

It is a **warning system**: nobody is kicked, and the joining player is never told. A flag
means a human should look — see [../ETHICS.md](../ETHICS.md) for why that boundary matters.

## Build

No JDK or Maven install needed; the build runs in a container.

```bash
./minecraft_plugin/build.sh          # -> minecraft_plugin/target/SkinBouncer.jar
./minecraft_plugin/build.sh clean    # extra goals are passed through to mvn
```

## Run

The quickest way is the demo compose file, which starts the API and a Paper server
together on one network:

```bash
EULA=TRUE MC_OPS=<your-minecraft-name> \
    docker compose -f 06_Deployment/docker-compose.demo.yml up --build
```

To use an existing server instead, drop `target/SkinBouncer.jar` into its `plugins/`
folder and point `api-url` in the generated `plugins/SkinBouncer/config.yml` at your API.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `api-url` | `http://api:8000/check/player/` | Where the detector API is reachable from the server process. |
| `timeout-seconds` | `30` | Whole-request budget. The API resolves the player through Mojang and downloads the skin before running inference. |
| `log-clean-joins` | `false` | Log a line for every join, not just flagged ones. Useful while setting things up. |

Staff warnings go to holders of `skinbouncer.notify` (default: op).

## Requirements

- **Paper 26.2** (Java 25). The server version in
  [`../06_Deployment/docker-compose.demo.yml`](../06_Deployment/docker-compose.demo.yml)
  is pinned to match the API this is compiled against — bump both together, or not at all.
- **The server must run in online mode.** The API identifies players through Mojang; an
  offline-mode server hands out UUIDs Mojang has never seen, so every lookup returns 404.
- **At least one exported detector**, otherwise the API scores nothing and no join is ever
  flagged. `GET /` lists what it loaded.

The HTTP call runs off the main thread — a join never waits on it, and an unreachable API
is logged rather than allowed to affect whether people can play.
