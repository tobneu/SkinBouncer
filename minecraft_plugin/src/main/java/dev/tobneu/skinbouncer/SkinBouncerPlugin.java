package dev.tobneu.skinbouncer;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.format.NamedTextColor;
import org.bukkit.Bukkit;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.plugin.java.JavaPlugin;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Scores every joining player's skin against a SkinBouncer detector API and warns staff
 * when a detector flags it.
 *
 * <p>This is a warning system: nobody is kicked and the joining player is never told.
 * A flag is a prompt for a human to look, which is the only use the model's precision
 * actually supports.
 */
public final class SkinBouncerPlugin extends JavaPlugin implements Listener {

    private static final String NOTIFY_PERMISSION = "skinbouncer.notify";

    private HttpClient http;
    private String apiUrl;
    private Duration timeout;
    private boolean logCleanJoins;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        apiUrl = getConfig().getString("api-url", "http://localhost:8000/check/player/");
        timeout = Duration.ofSeconds(getConfig().getInt("timeout-seconds", 30));
        logCleanJoins = getConfig().getBoolean("log-clean-joins", false);
        http = HttpClient.newBuilder().connectTimeout(timeout).build();
        getServer().getPluginManager().registerEvents(this, this);
        getLogger().info("Checking joining players against " + apiUrl);
    }

    @EventHandler
    public void onPlayerJoin(PlayerJoinEvent event) {
        final String playerName = event.getPlayer().getName();
        // The API resolves the name through Mojang and downloads the skin before it runs
        // inference. Holding the main thread for that would freeze the whole server for
        // the duration of every join.
        getServer().getScheduler().runTaskAsynchronously(this, () -> checkSkin(playerName));
    }

    private void checkSkin(String playerName) {
        JsonObject body = new JsonObject();
        body.addProperty("player_name", playerName);

        HttpRequest request = HttpRequest.newBuilder(URI.create(apiUrl))
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body.toString(), StandardCharsets.UTF_8))
                .build();

        HttpResponse<String> response;
        try {
            response = http.send(request, HttpResponse.BodyHandlers.ofString());
        } catch (Exception e) {
            // A moderation aid being unreachable must never affect whether people can
            // play, so this stays a log line rather than anything the player notices.
            getLogger().warning("Skin check for " + playerName + " failed: " + e.getMessage());
            return;
        }

        if (response.statusCode() == 404) {
            // The player has no custom skin, or Mojang could not resolve them. Nothing
            // to score, and not an error worth alarming an operator about.
            if (logCleanJoins) {
                getLogger().info("No skin to check for " + playerName);
            }
            return;
        }
        if (response.statusCode() != 200) {
            getLogger().warning("Skin check for " + playerName + " returned HTTP "
                    + response.statusCode() + ": " + response.body());
            return;
        }

        report(playerName, flaggedCategories(response.body(), playerName));
    }

    /** Returns "category (score)" for every detector that flagged this skin. */
    private List<String> flaggedCategories(String responseBody, String playerName) {
        List<String> flagged = new ArrayList<>();
        JsonObject categories;
        try {
            categories = JsonParser.parseString(responseBody)
                    .getAsJsonObject()
                    .getAsJsonObject("categories");
        } catch (Exception e) {
            getLogger().warning("Unreadable response for " + playerName + ": " + responseBody);
            return flagged;
        }

        for (Map.Entry<String, com.google.gson.JsonElement> entry : categories.entrySet()) {
            JsonObject result = entry.getValue().getAsJsonObject();
            double score = result.get("score").getAsDouble();
            if (result.get("risk").getAsBoolean()) {
                flagged.add(String.format("%s (%.2f)", entry.getKey(), score));
            } else if (logCleanJoins) {
                getLogger().info(String.format("%s: %s scored %.2f, below threshold",
                        playerName, entry.getKey(), score));
            }
        }
        return flagged;
    }

    private void report(String playerName, List<String> flagged) {
        if (flagged.isEmpty()) {
            if (logCleanJoins) {
                getLogger().info(playerName + ": no detector flagged this skin");
            }
            return;
        }

        String summary = String.join(", ", flagged);
        getLogger().warning("FLAGGED " + playerName + ": " + summary);

        Component message = Component.text("[SkinBouncer] ", NamedTextColor.GOLD)
                .append(Component.text(playerName, NamedTextColor.WHITE))
                .append(Component.text("'s skin was flagged: ", NamedTextColor.GRAY))
                .append(Component.text(summary, NamedTextColor.RED));

        // Adventure's broadcast is thread-safe, but scheduling it keeps every path that
        // touches players on the main thread, which is the rule worth not making
        // exceptions to.
        getServer().getScheduler().runTask(this,
                () -> Bukkit.broadcast(message, NOTIFY_PERMISSION));
    }
}
