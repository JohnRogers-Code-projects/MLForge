import { test, expect } from "@playwright/test";

test.describe("Jobs Page", () => {
  test("redirects to login when not authenticated", async ({ page }) => {
    await page.goto("/jobs");

    // Should be redirected to login
    await expect(page).toHaveURL(/\/login/);
  });
});
