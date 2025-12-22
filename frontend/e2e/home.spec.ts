import { test, expect } from "@playwright/test";

test.describe("Home Page", () => {
  test("home page renders correctly", async ({ page }) => {
    await page.goto("/");

    // Check main heading
    await expect(page.getByRole("heading", { name: "ModelForge", level: 1 })).toBeVisible();

    // Check tagline
    await expect(page.getByText("ML Model Serving Platform")).toBeVisible();

    // Check description
    await expect(
      page.getByText("Deploy, manage, and serve machine learning models with ease.")
    ).toBeVisible();

    // Check feature cards
    await expect(page.getByRole("heading", { name: "Models" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Predictions" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Jobs" })).toBeVisible();

    // Check feature descriptions
    await expect(page.getByText(/Upload and manage ONNX models/)).toBeVisible();
    await expect(page.getByText(/Run inference with caching/)).toBeVisible();
    await expect(page.getByText(/Async job processing/)).toBeVisible();
  });

  test("home page shows sign in button for unauthenticated users", async ({ page }) => {
    await page.goto("/");

    // Wait for auth state to resolve (loading state shows a placeholder)
    // The sign in button appears after the session is checked
    const signInButton = page.getByRole("button", { name: /Sign in/i });
    await expect(signInButton).toBeVisible({ timeout: 10000 });

    // Check that authenticated content is not visible
    await expect(page.getByText(/Welcome back/)).not.toBeVisible();
    await expect(page.getByRole("link", { name: "View Models" })).not.toBeVisible();
  });

  test("header navigation is visible", async ({ page }) => {
    await page.goto("/");

    // Check header logo/link
    const headerLink = page.locator("header").getByRole("link", { name: "ModelForge" });
    await expect(headerLink).toBeVisible();
  });

  test("API base URL is displayed", async ({ page }) => {
    await page.goto("/");

    // Check API URL is displayed
    await expect(page.getByText(/API:/)).toBeVisible();
  });
});
