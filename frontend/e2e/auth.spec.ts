import { test, expect } from "@playwright/test";

test.describe("Authentication", () => {
  test("login page is accessible", async ({ page }) => {
    await page.goto("/login");

    // Page should load without crashing
    // Note: In test environment without NEXTAUTH_SECRET, we may see error state
    await expect(page).toHaveURL(/\/login/);
  });

  test("login page shows error message when auth fails", async ({ page }) => {
    await page.goto("/login?error=OAuthAccountNotLinked");

    // Should show the error message
    await expect(
      page.getByText("This email is already associated with another account.")
    ).toBeVisible();
  });

  test("login page shows generic error for unknown errors", async ({ page }) => {
    await page.goto("/login?error=unknown");

    // Should show the generic error message
    await expect(
      page.getByText("Authentication failed. Please try again.")
    ).toBeVisible();
  });

  test("protected routes redirect to login", async ({ page }) => {
    // Try to access protected route
    await page.goto("/models");

    // Should be redirected to login
    await expect(page).toHaveURL(/\/login/);
  });

  test("jobs page redirects to login when not authenticated", async ({ page }) => {
    await page.goto("/jobs");

    // Should be redirected to login
    await expect(page).toHaveURL(/\/login/);
  });

  test("predictions page redirects to login when not authenticated", async ({ page }) => {
    await page.goto("/predictions");

    // Should be redirected to login
    await expect(page).toHaveURL(/\/login/);
  });
});
