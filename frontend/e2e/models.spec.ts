import { test, expect } from "@playwright/test";

test.describe("Models Page", () => {
  test("redirects to login when not authenticated", async ({ page }) => {
    await page.goto("/models");

    // Should be redirected to login
    await expect(page).toHaveURL(/\/login/);
  });

  test("redirects to login for model detail page", async ({ page }) => {
    await page.goto("/models/some-model-id");

    // Should be redirected to login
    await expect(page).toHaveURL(/\/login/);
  });

  test("redirects to login for new model page", async ({ page }) => {
    await page.goto("/models/new");

    // Should be redirected to login
    await expect(page).toHaveURL(/\/login/);
  });

  test("redirects to login for predict page", async ({ page }) => {
    await page.goto("/models/some-model-id/predict");

    // Should be redirected to login
    await expect(page).toHaveURL(/\/login/);
  });

  test("redirects to login for predictions history page", async ({ page }) => {
    await page.goto("/models/some-model-id/predictions");

    // Should be redirected to login
    await expect(page).toHaveURL(/\/login/);
  });
});
