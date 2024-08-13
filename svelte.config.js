// import adapter from '@sveltejs/adapter-auto';
import adapter from '@sveltejs/adapter-static';
// import adapter from '@sveltejs/adapter-cloudflare';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://kit.svelte.dev/docs/integrations#preprocessors
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: null,
			precompress: false,
			strict: false // Ignore errors about dynamic routes
		}),
		prerender: {
			// List the specific routes to prerender
			entries: ['/' /* other routes if needed */]
		},
		alias: {
			'~': './src'
		},
		paths: {
			base: process.env.NODE_ENV === 'production' ? '/transformer-explainer' : ''
		}
	}
};

export default config;
